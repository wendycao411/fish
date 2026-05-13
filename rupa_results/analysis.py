import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_localization_by_location():
    """
    Create a bar chart of F1 scores by location from localization_controls.csv,
    with baseline F1 as a horizontal line per location.
    """
    # Read CSV
    csv_path = Path(__file__).parent / "localization_controls.csv"
    df = pd.read_csv(csv_path)
    
    # Compute F1 from precision and recall
    df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"])
    
    # Baseline F1 values per location
    baseline_map = {
        "Danger Rocks": 0.379514719,
        "Taylor Islet": 0.3991304348,
    }
    
    # Get unique locations
    locations = sorted(df["location"].unique())
    
    # Create subplots for each location
    fig, axes = plt.subplots(1, len(locations), figsize=(5 * len(locations), 5), sharey=True)
    if len(locations) == 1:
        axes = [axes]
    
    for ax, loc in zip(axes, locations):
        sub = df[df["location"] == loc].copy()
        
        # Group by condition and take mean F1
        cond_f1 = sub.groupby("condition_id")["f1"].mean().reset_index()
        cond_f1 = cond_f1.sort_values("f1")
        
        # Bar chart
        bars = ax.bar(range(len(cond_f1)), cond_f1["f1"].values, color="#1f77b4", alpha=0.7)
        ax.set_xticks(range(len(cond_f1)))

        # Map condition IDs to presentation labels (wrap long names on two lines)
        label_map = {
            "random_heatmap": "Random\nheatmap",
            "shuffled_heatmap": "Shuffled\nheatmap",
            "true_heatmap": "True\nheatmap",
            "oracle_roi": "Oracle\nROI",
        }
        labels = [label_map.get(c, str(c).replace("_", "\n")) for c in cond_f1["condition_id"].values]
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
        
        # Baseline line
        baseline = baseline_map.get(loc)
        if baseline is not None:
            ax.axhline(baseline, color="red", linestyle="--", linewidth=2, label=f"Visual-only baseline: {baseline:.3f}")
        
        ax.set_title(f"{loc}")
        ax.set_ylabel("F1 score")
        ax.set_ylim(0, 1)
        ax.legend()
    
    plt.tight_layout()
    out_file = Path(__file__).parent / "localization_by_location.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_methods_comparison():
    """
    Grouped bar charts of precision and recall for:
    - visual_only_yolo (visual-only baseline)
    - motion_only (motion-only baseline)
    - late_fusion_audio_box (late-fusion)
    - audio_modulated_yolo (proposed method)
    
    One subplot per location (Danger Rocks, Taylor Islet).
    """
    csv_path = Path(__file__).parent / "precision_recall_by_location.csv"
    df = pd.read_csv(csv_path)
    
    # Select methods of interest
    methods_of_interest = [
        "visual_only_yolo",
        "motion_only",
        "late_fusion_audio_box",
        "audio_modulated_yolo"
    ]
    df = df[df["method_id"].isin(methods_of_interest)]
    
    # Get unique locations
    locations = sorted(df["location_id"].unique())
    
    # Create subplots
    fig, axes = plt.subplots(1, len(locations), figsize=(6 * len(locations), 5), sharey=True)
    if len(locations) == 1:
        axes = [axes]
    
    for ax, loc in zip(axes, locations):
        sub = df[df["location_id"] == loc].copy()
        
        # Get data for each method in order
        methods = []
        precisions = []
        recalls = []
        for method in methods_of_interest:
            row = sub[sub["method_id"] == method]
            if not row.empty:
                methods.append(method)
                precisions.append(row["precision"].values[0])
                recalls.append(row["recall"].values[0])
        
        # Bar positions
        x = np.arange(len(methods))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, precisions, width, label="Precision", color="#1f77b4", alpha=0.8)
        bars2 = ax.bar(x + width/2, recalls, width, label="Recall", color="#ff7f0e", alpha=0.8)
        
        # Labels and formatting
        ax.set_ylabel("Score")
        ax.set_title(f"Location: {loc}")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    out_file = Path(__file__).parent / "methods_comparison.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_alpha_ablation():
    """
    Line plots of precision, recall, and F1 vs alpha for each location.
    One separate figure per location (Danger Rocks, Taylor Islet).
    """
    csv_path = Path(__file__).parent / "alpha_ablation_v2.csv"
    df = pd.read_csv(csv_path)
    
    # Compute F1
    df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"])
    
    # Get unique locations
    locations = sorted(df["location_id"].unique())
    
    for loc_id in locations:
        sub = df[df["location_id"] == loc_id].copy()
        sub = sub.sort_values("alpha")
        
        # Get location display name
        loc_name = sub["location"].iloc[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot lines
        ax.plot(sub["alpha"], sub["precision"], marker="o", label="Precision", color="#1f77b4", linewidth=2)
        ax.plot(sub["alpha"], sub["recall"], marker="o", label="Recall", color="#ff7f0e", linewidth=2)
        ax.plot(sub["alpha"], sub["f1"], marker="s", label="F1", color="#2ca02c", linewidth=2.5)
        
        # Mark optimal alpha (highest F1)
        best_idx = sub["f1"].idxmax()
        best_alpha = sub.loc[best_idx, "alpha"]
        best_f1 = sub.loc[best_idx, "f1"]
        ax.scatter([best_alpha], [best_f1], s=150, color="#2ca02c", zorder=5, edgecolors="black", linewidth=1.5)
        
        # Labels and formatting
        ax.set_xlabel("Alpha (audio weighting parameter)")
        ax.set_ylabel("Score")
        ax.set_title(f"{loc_name}")
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_file = Path(__file__).parent / f"alpha_ablation_{loc_id}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")


def plot_labeled_data_scaling():
    """
    Line plots of precision, recall, and F1 vs labeled_frames for each location.
    X-axis uses log scale to show saturation point.
    One separate figure per location (Danger Rocks, Taylor Islet).
    """
    csv_path = Path(__file__).parent / "labeled_data_scaling.csv"
    df = pd.read_csv(csv_path)
    
    # Get unique locations
    locations = sorted(df["location_id"].unique())
    
    for loc_id in locations:
        sub = df[df["location_id"] == loc_id].copy()
        sub = sub.sort_values("labeled_frames")
        
        # Get location display name
        loc_name = sub["location"].iloc[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot lines with log-scale X-axis
        ax.plot(sub["labeled_frames"], sub["precision"], marker="o", label="Precision", color="#1f77b4", linewidth=2)
        ax.plot(sub["labeled_frames"], sub["recall"], marker="o", label="Recall", color="#ff7f0e", linewidth=2)
        ax.plot(sub["labeled_frames"], sub["f1"], marker="s", label="F1", color="#2ca02c", linewidth=2.5)
        
        # Use log scale for X-axis
        ax.set_xscale("log")
        
        # Labels and formatting
        ax.set_xlabel("Number of labeled frames")
        ax.set_ylabel("Score")
        ax.set_title(f"{loc_name}")
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        
        plt.tight_layout()
        out_file = Path(__file__).parent / f"labeled_data_scaling_{loc_id}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")


def plot_nighttime_f1_by_method():
    """
    Bar charts comparing F1 scores (daytime vs nighttime) for each method.
    One figure per location (Danger Rocks, Taylor Islet).
    Methods: Visual-only YOLO, Motion-only, Late fusion audio box, 
             Heatmap-channel YOLO, Audio-modulated YOLO
    """
    day_csv = Path(__file__).parent / "precision_recall_by_location.csv"
    night_csv = Path(__file__).parent / "nighttime_precision_recall_by_location.csv"
    
    df_day = pd.read_csv(day_csv)
    df_night = pd.read_csv(night_csv)
    
    # Compute F1
    df_day["f1"] = 2 * df_day["precision"] * df_day["recall"] / (df_day["precision"] + df_day["recall"])
    df_night["f1"] = 2 * df_night["precision"] * df_night["recall"] / (df_night["precision"] + df_night["recall"])
    
    # Method display names mapping
    method_display = {
        "visual_only_yolo": "Visual-only\nYOLO",
        "motion_only": "Motion-only",
        "late_fusion_audio_box": "Late fusion\naudio box",
        "heatmap_channel_yolo": "Heatmap-channel\nYOLO",
        "audio_modulated_yolo": "Audio-modulated\nYOLO",
    }
    
    # Methods of interest in order
    methods_of_interest = [
        "visual_only_yolo",
        "motion_only",
        "late_fusion_audio_box",
        "heatmap_channel_yolo",
        "audio_modulated_yolo",
    ]
    
    # Map location_id to display name
    loc_map = {
        "danger_rocks": "Danger Rocks",
        "taylor_islet": "Taylor Islet",
    }
    
    # Get unique locations
    locations = sorted(df_day["location_id"].unique())
    
    for loc_id in locations:
        day_sub = df_day[df_day["location_id"] == loc_id].copy()
        night_sub = df_night[df_night["location_id"] == loc_id].copy()
        
        # Get location display name
        loc_name = loc_map.get(loc_id, loc_id)
        
        # Prepare data for each method
        day_f1s = []
        night_f1s = []
        method_labels = []
        for m in methods_of_interest:
            day_f1 = day_sub[day_sub["method_id"] == m]["f1"].values[0] if (day_sub["method_id"] == m).any() else 0
            night_f1 = night_sub[night_sub["method_id"] == m]["f1"].values[0] if (night_sub["method_id"] == m).any() else 0
            day_f1s.append(day_f1)
            night_f1s.append(night_f1)
            method_labels.append(method_display.get(m, m))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(method_labels))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, day_f1s, width, label="Daytime", color="#1f77b4", alpha=0.8)
        bars2 = ax.bar(x + width/2, night_f1s, width, label="Nighttime", color="#ff7f0e", alpha=0.8)
        
        # Labels and formatting
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{loc_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=0, ha="center", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        out_file = Path(__file__).parent / f"f1_by_method_{loc_id}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_localization_by_location()
    plot_methods_comparison()
    plot_alpha_ablation()
    plot_labeled_data_scaling()
    plot_nighttime_f1_by_method()

