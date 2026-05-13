#!/usr/bin/env python3
"""Build a balanced YOLO dataset split between Taylor Islet and Danger Rocks.

Produces a dataset directory with `images/train` (symlinks) and `labels/train` (copied labels
or symlinked when available), plus `data.yaml` and `train.txt`.

Sampling rules (configurable):
- target_total: desired number of images (approx)
- taylor_share: fraction of images from Taylor Islet (rest from Danger)
- localization_share: fraction of images that are at exact localization (i.e., have labels)
- min_frame_gap: prefer selecting frames separated by at least this many frames within same video

Workflow:
1. Index all .jpg images under `search_root` and build a basename->path map.
2. Index Taylor label files (from provided LABEL_SEARCH_DIRS) -> localization pool.
3. Index Danger localization label files under `danger_labels/labels/train` -> localization pool.
4. Create non-localization candidate pool by taking images not in the localization pool.
5. Sample per-video while enforcing `min_frame_gap` where possible to avoid consecutive images.
6. Create output directory and symlink chosen images; copy/symlink labels for localized images when available.

This script is conservative (uses symlinks by default) to avoid duplicating large image files.

Run: python darienne_scripts/build_balanced_yolo_dataset.py --help
"""

import argparse
import os
import shutil
import random
import re
from collections import defaultdict


def find_images(search_root, exts=(".jpg", ".jpeg", ".png")):
    images = {}
    for root, _, files in os.walk(search_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                base = os.path.splitext(f)[0]
                images[base] = os.path.join(root, f)
    return images


def index_label_files(label_dirs):
    label_index = {}
    for d in label_dirs:
        if not os.path.exists(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith('.txt'):
                base = os.path.splitext(fn)[0]
                label_index[base] = os.path.join(d, fn)
    return label_index


def extract_video_and_frame(base_name):
    # Attempt to parse video id and frame index from names like: 5064_FishCam03_..._frame0046_...
    m_vid = re.match(r'^(\d+)_', base_name)
    vid = m_vid.group(1) if m_vid else None

    m_frame = re.search(r'frame(\d+)', base_name)
    frame = int(m_frame.group(1)) if m_frame else None
    return vid, frame


def group_by_video(basenames):
    by_vid = defaultdict(list)
    for b in basenames:
        vid, frame = extract_video_and_frame(b)
        key = vid or 'unknown'
        by_vid[key].append((b, frame))
    # sort by frame where possible
    for k, vals in by_vid.items():
        by_vid[k] = sorted(vals, key=lambda x: (x[1] is None, x[1] or 0))
    return by_vid


def spaced_sample_from_video_group(group, n, min_gap):
    # group is list of (basename, frame) sorted by frame; try greedy sampling enforcing min_gap
    chosen = []
    last_frame = -10**9
    for base, frame in group:
        f = frame if frame is not None else None
        if f is None or f - last_frame >= min_gap:
            chosen.append(base)
            last_frame = f if f is not None else last_frame
            if len(chosen) >= n:
                break
    return chosen


def sample_pool(local_taylor, local_danger, nonlocal_pool, total, taylor_share, localization_share, min_gap):
    n_taylor = int(round(total * taylor_share))
    n_danger = total - n_taylor

    n_local = int(round(total * localization_share))
    n_nonlocal = total - n_local

    # allocate localization samples proportionally across taylor/danger
    n_local_taylor = int(round(n_local * taylor_share))
    n_local_danger = n_local - n_local_taylor

    taylor_selected = set()
    danger_selected = set()

    # Sample localized Taylor and Danger
    taylor_local_list = list(local_taylor)
    danger_local_list = list(local_danger)
    random.shuffle(taylor_local_list)
    random.shuffle(danger_local_list)

    taylor_selected.update(taylor_local_list[:n_local_taylor])
    danger_selected.update(danger_local_list[:n_local_danger])

    # For remaining slots, sample non-local images, trying to maintain spacing per-video
    remaining_taylor = n_taylor - len(taylor_selected)
    remaining_danger = n_danger - len(danger_selected)

    # Group candidates by video to ensure spacing
    nonlocal_by_vid = group_by_video(nonlocal_pool)
    # Build flattened candidate lists per domain by checking basename source path contains key words
    all_nonlocal_bases = list(nonlocal_pool)
    random.shuffle(all_nonlocal_bases)

    # simple greedy fill that avoids choosing adjacent frames within min_gap for same video
    def pick_nonlocal_for_domain(k):
        picked = set()
        need = k
        # iterate videos in shuffled order
        vids = list(nonlocal_by_vid.keys())
        random.shuffle(vids)
        for vid in vids:
            group = nonlocal_by_vid[vid]
            # how many from this group? pick 1 at most in first pass
            if need <= 0:
                break
            sel = spaced_sample_from_video_group(group, 1, min_gap)
            if sel:
                picked.add(sel[0])
                need -= 1
        # if still need, fill from remaining pool without spacing
        if need > 0:
            for b in all_nonlocal_bases:
                if b in picked:
                    continue
                picked.add(b)
                need -= 1
                if need <= 0:
                    break
        return picked

    taylor_selected.update(pick_nonlocal_for_domain(remaining_taylor))
    danger_selected.update(pick_nonlocal_for_domain(remaining_danger))

    # final trimming if over
    def trim_set(s, k):
        if len(s) <= k:
            return set(s)
        return set(random.sample(list(s), k))

    taylor_selected = trim_set(taylor_selected, n_taylor)
    danger_selected = trim_set(danger_selected, n_danger)

    return taylor_selected, danger_selected


def write_dataset_split(output_dir, images_map, labels_map, train_set, test_set, symlink=True):
    images_train = os.path.join(output_dir, 'images', 'train')
    images_test = os.path.join(output_dir, 'images', 'test')
    labels_train = os.path.join(output_dir, 'labels', 'train')
    labels_test = os.path.join(output_dir, 'labels', 'test')
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(images_test, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_test, exist_ok=True)

    def place(base, dst_img_dir, dst_label_dir):
        src = images_map.get(base)
        if not src or not os.path.exists(src):
            return None
        dst = os.path.join(dst_img_dir, base + os.path.splitext(src)[1])
        try:
            if symlink:
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception:
            try:
                shutil.copy2(src, dst)
            except Exception:
                return None

        lab = labels_map.get(base)
        if lab and os.path.exists(lab):
            dstlab = os.path.join(dst_label_dir, base + '.txt')
            try:
                if symlink:
                    if os.path.exists(dstlab):
                        os.remove(dstlab)
                    os.symlink(lab, dstlab)
                else:
                    shutil.copy2(lab, dstlab)
            except Exception:
                try:
                    shutil.copy2(lab, dstlab)
                except Exception:
                    pass
        return os.path.join(os.path.basename(dst_img_dir), os.path.basename(dst))

    train_list = []
    for base in sorted(train_set):
        rel = place(base, images_train, labels_train)
        if rel:
            train_list.append(os.path.join('images', 'train', rel.split(os.sep)[-1]))

    test_list = []
    for base in sorted(test_set):
        rel = place(base, images_test, labels_test)
        if rel:
            test_list.append(os.path.join('images', 'test', rel.split(os.sep)[-1]))

    train_txt = os.path.join(output_dir, 'train.txt')
    with open(train_txt, 'w') as f:
        for p in sorted(train_list):
            f.write(p + '\n')

    val_txt = os.path.join(output_dir, 'val.txt')
    with open(val_txt, 'w') as f:
        for p in sorted(test_list):
            f.write(p + '\n')

    data_yaml = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml, 'w') as f:
        f.write('names:\n  0: fish\npath: .\ntrain: train.txt\nval: val.txt\n')

    return len(train_list), len(test_list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--search-root', default='.', help='Root to search images under (default repo root)')
    p.add_argument('--taylor-label-dirs', nargs='+', default=[
        '/Users/wendycao/fish/taylor_labels/batch1/labels/train',
        '/Users/wendycao/fish/taylor_labels/batch2/labels/train',
        '/Users/wendycao/fish/taylor_labels/batch3/labels/train',
        '/Users/wendycao/fish/taylor_labels/batch4/labels/train',
        '/Users/wendycao/fish/taylor_labels/batch_001/labels/train',
    ])
    p.add_argument('--danger-label-dir', default='danger_labels/labels/train')
    p.add_argument('--output-dir', default='balanced_yolo_dataset')
    p.add_argument('--total', type=int, default=10000)
    p.add_argument('--taylor-share', type=float, default=0.7)
    p.add_argument('--localization-share', type=float, default=0.6)
    p.add_argument('--min-gap', type=int, default=20, help='Minimum frame gap to prefer within same video')
    p.add_argument('--train-ratio', type=float, default=0.7, help='Fraction of selected images placed in train (rest in val)')
    p.add_argument('--no-symlink', dest='symlink', action='store_false')
    p.set_defaults(symlink=True)
    args = p.parse_args()

    print('Indexing images under', args.search_root)
    images_map = find_images(args.search_root)
    print('  Found', len(images_map), 'images')

    print('Indexing Taylor labels...')
    taylor_labels = index_label_files(args.taylor_label_dirs)
    print('  Found', len(taylor_labels), 'Taylor label files')

    print('Indexing Danger labels...')
    danger_label_dir = args.danger_label_dir
    danger_labels = {}
    if os.path.exists(danger_label_dir):
        danger_labels = index_label_files([danger_label_dir])
    else:
        # try to find within repo path
        if os.path.exists(os.path.join('.', danger_label_dir)):
            danger_labels = index_label_files([os.path.join('.', danger_label_dir)])

    print('  Found', len(danger_labels), 'Danger label files')

    # Build localization base name sets
    taylor_local_bases = set(b for b in taylor_labels.keys() if b in images_map)
    danger_local_bases = set(b for b in danger_labels.keys() if b in images_map)

    print(f'  Taylor localized available: {len(taylor_local_bases)}')
    print(f'  Danger localized available: {len(danger_local_bases)}')

    # Non-localization pool: all images excluding localized
    localized_all = set(list(taylor_local_bases) + list(danger_local_bases))
    nonlocal_bases = set(images_map.keys()) - localized_all
    print('  Non-localization candidates:', len(nonlocal_bases))

    total = args.total
    print(f'Sampling approx {total} images with taylor_share={args.taylor_share}, localization_share={args.localization_share}')

    taylor_sel, danger_sel = sample_pool(
        taylor_local_bases, danger_local_bases, nonlocal_bases,
        total, args.taylor_share, args.localization_share, args.min_gap
    )

    print('Selected: Taylor', len(taylor_sel), 'Danger', len(danger_sel), 'Total', len(taylor_sel) + len(danger_sel))

    # split each domain selection into train/test while preserving domain proportions
    def split_set(s, train_ratio):
        s_list = list(s)
        random.shuffle(s_list)
        n_train = int(round(len(s_list) * train_ratio))
        return set(s_list[:n_train]), set(s_list[n_train:])

    taylor_train, taylor_test = split_set(taylor_sel, args.train_ratio)
    danger_train, danger_test = split_set(danger_sel, args.train_ratio)

    train_set = taylor_train.union(danger_train)
    test_set = taylor_test.union(danger_test)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    n_train, n_test = write_dataset_split(out, images_map, {**taylor_labels, **danger_labels}, train_set, test_set, symlink=args.symlink)

    print('Wrote dataset to', out)
    print('  Images (symlinked): train=', n_train, 'val=', n_test)
    print('  train.txt:', os.path.join(out, 'train.txt'))
    print('  val.txt:', os.path.join(out, 'val.txt'))
    print('  data.yaml:', os.path.join(out, 'data.yaml'))


if __name__ == '__main__':
    main()
