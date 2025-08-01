import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import fire
import seaborn as sns
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator 
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter

# Plotting format
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.75,
        "lines.markersize": 4,
        "axes.edgecolor": "black",
        "axes.facecolor": "#f7f7f7",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "gray",
        "legend.framealpha": 1.0,
        "savefig.dpi": 300,
    })

CMAP = {
    "Smol135M-weight":  "#7ad151",  # bright green
    "Smol135M-tool":    "#7ad151",

    "Smol360M-weight":  "#22a884",  # viridis green
    "Smol360M-tool":    "#22a884",

    "Lam1B-weight":     "#2a788e",  # teal-blue
    "Lam1B-tool":       "#2a788e",

    "Smol1.7B-weight":  "#414487",  # violet
    "Smol1.7B-tool":    "#414487",

    "Lam3B-weight":     "#440154",  # deep purple
    "Lam3B-tool":       "#440154",

    "Lam8B-weight":     "#270031",  # black
    "Lam8B-tool":       "#270031",
}

# =================================================
# ==========    Plotting Functions    =============
# =================================================

# ----------    HellaSwag Plots    ------------

# ---- Plot Hellaswag_performace (absolute) vs Dataset_size
def plot_hellaswag_vs_datasetsize_absolute(df, save_path, acc_threshold, save_name="hellaswag_vs_facts_absolute"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5/1.3)) 

    size_map = {
        "0": 0, "500": 500, "1k": 1000, "5k": 5000, "10k": 10000, "50k": 50000, "all": 500
    }
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    model_labels = [
        "Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"
    ]

    color_map = CMAP

    marker_style = {
        "Lam1B-weight": 'o', "Lam3B-weight": 'o', "Lam8B-weight": 'o',
        "Smol135M-weight": '^', "Smol360M-weight": '^', "Smol1.7B-weight": '^',
    }

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }

    # Plotting x-axis scale:
    size_map = {
        "0": 350,       # fake-log position for 0
        "500": 512,
        "1k": 1024,
        "5k": 2096,
        "10k": 8192,
        "50k": 50000,
        "all": 512  # map to same as 500 for consistency
    }

    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    # Keep only last checkpoint per model/dataset
    df = df.sort_values("checkpoint_nbr").groupby(["model", "dataset_size"], as_index=False).last()
    for model_label in sorted(df["model"].unique()):
        sub = df[df["model"] == model_label].sort_values("dataset_size_n")
        color = color_map[model_label]
        label = labels[model_label]

        if "tool" in model_label:
            # Horizontal dashed line with shaded area
            x_vals = [350, 500, 1000, 5000, 10000, 50000]
            #x_vals = sub["dataset_size_n"].values
            y_mean = sub["hellaswag_absolute"].mean() * 100
            y_std = sub["hellaswag_absolute_stderr"].mean() * 100
            ax.plot(x_vals, [y_mean] * len(x_vals), linestyle="--", color=color, label=label, linewidth=1)
            ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)
        else:
            x = sub["dataset_size_n"].values
            y = sub["hellaswag_absolute"].values * 100
            yerr = sub["hellaswag_absolute_stderr"].values * 100
            ax.plot(x, y, color=color, label=label, marker=marker_style.get(model_label, 'o')) # linewidth=1.75
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    # #ax.set_xscale("log")
    # ax.set_xticks([0, 500, 1000, 5000, 10000, 50000])
    # ax.set_xticklabels(["0", "500", "1k", "5k", "10k", "50k"])
    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1K", "5K", "10K", "50K"])
    percent_str = f"{int(acc_threshold * 100)}%"
    ax.set_xlabel(f"Facts Memorized (≥{percent_str} Recall)")
    ax.set_ylabel("HellaSwag Accuracy (%)")

    # Legend for line style (in-weight vs in-tool)
    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle='-', linewidth=1.5, label="In-Weight"),
            Line2D([0], [0], color="black", linestyle='--', linewidth=1.5, label="In-Tool"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.02),
        ncol=2,
        frameon=False, #False
        handletextpad=0.25,
        columnspacing=1.,
        handlelength=1.35,
    )
    ax.add_artist(linestyle_legend)

    # Family legend with model sizes
    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4,  label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-',linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="1.7B"),
    ]

    family_legend = ax.legend(
        handles=llama_handles + smol_handles,
        labels=["1B", "3B", "8B     ", "135M", "360M", "1.7B"],
        loc="upper center",
        bbox_to_anchor=(0.495, 1.146),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        handlelength=1.3,
        fontsize=7.6
    )

    ax.add_artist(family_legend)

    # Add manual group titles for families
    ax.text(0.205, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.738, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{acc_threshold}.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{acc_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ---- Plot Hellaswag_performace (absolute) vs Dataset_size
def plot_hellaswag_vs_datasetsize_relative(df, save_path, acc_threshold, save_name="hellaswag_vs_facts_relative"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5/1.3)) 

    size_map = {
        "0": 0, "500": 500, "1k": 1000, "5k": 5000, "10k": 10000, "50k": 50000, "all": 500
    }
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    model_labels = [
        "Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"
    ]
   
    marker_style = {
        "Lam1B-weight": 'o', "Lam3B-weight": 'o', "Lam8B-weight": 'o',
        "Smol135M-weight": '^', "Smol360M-weight": '^', "Smol1.7B-weight": '^',
    }

    color_map = CMAP

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }

    # Plotting x-axis scale:
    size_map = {
        "0": 350,       # fake-log position for 0
        "500": 512,
        "1k": 1024,
        "5k": 2096,
        "10k": 8192,
        "50k": 50000,
        "all": 512  # map to same as 500 for consistency
    }
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    # Keep only last checkpoint per model/dataset
    tool_baseline_plotted = False
    df = df.sort_values("checkpoint_nbr").groupby(["model", "dataset_size"], as_index=False).last()
    for model_label in sorted(df["model"].unique()):
        sub = df[df["model"] == model_label].sort_values("dataset_size_n")
        color = color_map[model_label]
        label = labels[model_label]
        
        if "tool" in model_label:
            if not tool_baseline_plotted: # Only plot the first since they are about the same
                # Horizontal dashed line with shaded area
                x_vals = [350, 500, 1000, 5000, 10000, 50000]
                #x_vals = sub["dataset_size_n"].values
                y_mean = sub["hellaswag_relative"].mean()
                y_std = sub["hellaswag_relative_stderr"].mean()
                ax.plot(x_vals, [y_mean] * len(x_vals), linestyle="--", color="black", label=label, linewidth=1)
                ax.fill_between(x_vals, y_mean - y_std, min(y_mean + y_std, 100), color="black", alpha=0.2)
                tool_baseline_plotted = True
            continue 
        else:
            x = sub["dataset_size_n"].values
            y = sub["hellaswag_relative"].values
            yerr = sub["hellaswag_relative_stderr"].values
            ax.plot(x, y, color=color, label=label, marker=marker_style.get(model_label, 'o')) # linewidth=1.75
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1K", "5K", "10K", "50K"])
    percent_str = f"{int(acc_threshold * 100)}%"
    ax.set_xlabel(f"Facts Memorized (≥{percent_str} Recall)") # Facts Memorized
    ax.set_ylabel("HellaSwag Performace", labelpad=13)
    ax.text(
        -0.12, 0.5,  # x = left of axis, y = centered vertically
        "(% relative to base model)",
        fontsize=8.5,
        rotation=90,
        va='center',
        ha='right',
        transform=ax.transAxes
    )

    # Legend for line style (in-weight vs in-tool)
    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle='-', linewidth=1.5, label="In-Weight"),
            Line2D([0], [0], color="black", linestyle='--', linewidth=1.5, label="In-Tool baseline"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.01),
        ncol=2,
        frameon=False, #False
        handletextpad=0.25,
        columnspacing=1.,
        handlelength=1.38,
    )
    ax.add_artist(linestyle_legend)

    # Family legend with model sizes
    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4,  label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-',linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="1.7B"),
    ]

    family_legend = ax.legend(
        handles=llama_handles + smol_handles,
        labels=["1B", "3B", "8B    ", "135M", "360M", "1.7B"],
        loc="upper center",
        bbox_to_anchor=(0.4945, 1.146),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        handlelength=1.3,
        fontsize=7.6
    )
    ax.add_artist(family_legend)

    # Add manual group titles for families
    ax.text(0.209, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.746, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{acc_threshold}.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{acc_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----------    Total Variation Plots    ------------

# ----- Plot TotalVariation vs Dataset_Size
def plot_final_tv_vs_dataset_size(df, save_path, recall_threshold=0.9, save_name="final_tv_vs_facts"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5 / 1.3))  # ICLR-style

    size_map = {
        "0": 0, "500": 512, "1k": 1024, "5k": 2096,
        "10k": 8192, "50k": 50000, "all": 512
    }
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    model_labels = [
        "Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"
    ]    

    marker_style = {
        "Lam1B-weight": 'o', "Lam3B-weight": 'o', "Lam8B-weight": 'o',
        "Smol135M-weight": '^', "Smol360M-weight": '^', "Smol1.7B-weight": '^',
    }

    color_map = CMAP

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }

    # Compute the highest tool baseline across all datasets
    tool_df = df[df["model"].str.contains("tool")]
    tool_baseline_y = tool_df["tv"].max()

    # Add horizontal tool baseline line across full x-range
    ax.axhline(y=tool_baseline_y, color="black", linestyle="--", linewidth=1, label="tool baseline")

    final_tv_df = df[df["recall"] >= recall_threshold].sort_values("checkpoint_nbr")
    final_tv_df = final_tv_df.groupby(["model", "dataset_size"], as_index=False).first()

    # Add (0, 0) point for each unique model
    unique_models = final_tv_df["model"].unique()
    zero_rows = []
    for model in unique_models:
        zero_rows.append({
            "model": model,
            "dataset_size": "0",
            "dataset_size_n": 350,  # fake 0 position for log-scale
            "tv": 0.0,
            "tv_stderr": 1e-8,
        })
    final_tv_df = pd.concat([pd.DataFrame(zero_rows), final_tv_df], ignore_index=True)
    final_tv_df = final_tv_df.sort_values(["model", "dataset_size_n"])

    for model_label in sorted(final_tv_df["model"].unique()):
        sub = final_tv_df[final_tv_df["model"] == model_label].sort_values("dataset_size_n")
        color = color_map[model_label]
        label = labels[model_label]

        if "tool" in model_label:
            continue
        else:
            x = sub["dataset_size_n"].values
            y = sub["tv"].values
            yerr = sub["tv_stderr"].values.copy()
            yerr = np.where(x == 350, 1e-8, yerr)  # Use handcrafted stderr for (0,0) only

            ax.plot(x, y, color=color, label=label, marker=marker_style.get(model_label, 'o'))
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
            
    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1K", "5K", "10K", "50K"])
    percent_str = f"{int(recall_threshold * 100)}%"
    ax.set_xlabel(f"Facts Memorized (≥{percent_str} Recall)")
    ax.set_ylabel("Total Variation")

    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle='-', linewidth=1.5, label="In-Weight"),
            Line2D([0], [0], color="black", linestyle='--', linewidth=1.5, label="In-Tool baseline (worst)"),
        ],
        loc="upper left",
        #bbox_to_anchor=(0.0, -0.02),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=1.,
        handlelength=1.35,
    )
    ax.add_artist(linestyle_legend)

    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4,  label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-',linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="1.7B"),
    ]
    family_legend = ax.legend(
        handles=llama_handles + smol_handles,
        labels=["1B", "3B", "8B     ", "135M", "360M", "1.7B"],
        loc="upper center",
        bbox_to_anchor=(0.495, 1.146),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        handlelength=1.3,
        fontsize=7.6
    )
    ax.add_artist(family_legend)
    ax.text(0.205, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.738, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{recall_threshold}.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{recall_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()

# ----- Plot TotalVariation vs Dataset_Size, log scale, with all tool runs
def plot_final_tv_vs_dataset_size_withtool(df, save_path, recall_threshold=0.9, save_name="final_tv_vs_facts"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5 / 1.3))  # ICLR-style

    size_map = {
        "0": 0, "500": 512, "1k": 1024, "5k": 2096,
        "10k": 8192, "50k": 50000, "all": 512
    }
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    model_labels = [
        "Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"
    ]   

    color_map = CMAP

    marker_style = {
        "Lam1B-weight": 'o', "Lam1B-tool": 'o',
        "Lam3B-weight": 'o', "Lam3B-tool": 'o',
        "Lam8B-weight": 'o', "Lam8B-tool": 'o',
        "Smol135M-weight": '^', "Smol135M-tool": '^',
        "Smol360M-weight": '^', "Smol360M-tool": '^',
        "Smol1.7B-weight": '^', "Smol1.7B-tool": '^',
    }

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }

    tool_df = df[df["model"].str.contains("tool")]
    tool_baseline_y = tool_df["tv"].max()
    #ax.axhline(y=tool_baseline_y, color="black", linestyle="--", linewidth=1, label="tool baseline")

    final_tv_df = df[df["recall"] >= recall_threshold].sort_values("checkpoint_nbr")
    final_tv_df = final_tv_df.groupby(["model", "dataset_size"], as_index=False).first()

    zero_rows = []
    for model in final_tv_df["model"].unique():
        zero_rows.append({
            "model": model,
            "dataset_size": "0",
            "dataset_size_n": 350,
            "tv": 1e-2,
            "tv_stderr": 1e-8,
        })

    tool_extensions = []
    dataset_sizes_for_extension = ["500", "1k", "5k", "10k", "50k"]
    for model in df["model"].unique():
        if "tool" not in model:
            continue
        tool_row = final_tv_df[final_tv_df["model"] == model].head(1)
        if tool_row.empty:
            continue
        tv_value = tool_row["tv"].values[0]
        stderr_value = tool_row.get("tv_stderr", pd.Series([0.0])).values[0]
        for size in dataset_sizes_for_extension:
            tool_extensions.append({
                "model": model,
                "dataset_size": size,
                "dataset_size_n": size_map[size],
                "tv": tv_value,
                "tv_stderr": stderr_value,
            })

    full_df = pd.concat([final_tv_df, pd.DataFrame(zero_rows + tool_extensions)], ignore_index=True)
    full_df = full_df.sort_values(["model", "dataset_size_n"])

    for idx, model_label in enumerate(sorted(full_df["model"].unique())):
        sub = full_df[full_df["model"] == model_label].sort_values("dataset_size_n")
        color = color_map[model_label]
        label = labels[model_label]

        x_vals = sub["dataset_size_n"].values
        y = sub["tv"].values
        yerr = sub["tv_stderr"].values.copy()
        yerr = np.where(x_vals == 350, 1e-8, yerr)  

        if "tool" in model_label:
            y[1:] = y[1:] - 0.0005 * (idx + 1) # vertical shift to distinguish between overlaps
            ax.plot(
                x_vals, y, 
                color=color, label=label,
                linestyle="--", marker=marker_style.get(model_label, 'o'),
                linewidth=1.25, markersize=3.7
            )
            ax.fill_between(x_vals, y - yerr, y + yerr, color=color, alpha=0.2)
            
        else:
            ax.plot(
                x_vals, y, color=color, label=label,
                linestyle="-", marker=marker_style.get(model_label, 'o'),
            )
            ax.fill_between(x_vals, y - yerr, y + yerr, color=color, alpha=0.2)


    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1K", "5K", "10K", "50K"])
    percent_str = f"{int(recall_threshold * 100)}%"
    ax.set_xlabel(f"Facts Memorized (≥{percent_str} Recall)")
    
    ax.set_ylabel("Total Variation")
    ax.set_yscale("log")
    tick_vals = [1e-2, 1.5e-2, 1e-1, 1]
    tick_labels = ['0', '0.01', '0.1', '1']
    ax.yaxis.set_major_locator(FixedLocator(tick_vals))
    ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))

    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle='-', linewidth=1.5, label="In-Weight"),
            Line2D([0], [0], color="black", linestyle='--', linewidth=1.5, label="In-Tool"),
        ],
        loc="upper left",
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=1.,
        handlelength=1.35,
    )
    ax.add_artist(linestyle_legend)

    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="1.7B"),
    ]
    family_legend = ax.legend(
        handles=llama_handles + smol_handles,
        labels=["1B", "3B", "8B     ", "135M", "360M", "1.7B"],
        loc="upper center",
        bbox_to_anchor=(0.495, 1.146),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        handlelength=1.3,
        fontsize=7.6
    )
    ax.add_artist(family_legend)
    ax.text(0.205, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.738, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{recall_threshold}_new.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{recall_threshold}_new.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----- Plot Total Variation vs Training Steps (all models, specific dataset_size)
def plot_tv_vs_training_steps(df, save_path, dataset_size="10k", recall_threshold=0.9, mode="TV", save_name=None):
    fig, ax = plt.subplots(figsize=(2.15, 2.15 / 1.2))  

    model_labels = [
        "Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"
    ]
    
    color_map = CMAP

    marker_style = {
        "Lam1B-weight": 'o', "Lam3B-weight": 'o', "Lam8B-weight": 'o',
        "Smol135M-weight": '^', "Smol360M-weight": '^', "Smol1.7B-weight": '^',
    }

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }

    df_subset = df[df["dataset_size"] == dataset_size].copy()
    df_subset = df_subset.sort_values("checkpoint_nbr")

    tool_df = df_subset[df_subset["model"].str.contains("tool")]
    if not tool_df.empty:
        tool_baseline_y = tool_df["tv"].max() if mode == "TV" else tool_df["kl"].max()
        ax.axhline(tool_baseline_y, color="black", linestyle="--", linewidth=1.05, label="tool baseline")

    for model_label in sorted(df_subset["model"].unique()):
        if "tool" in model_label:
            continue

        sub = df_subset[df_subset["model"] == model_label]
        sub = sub[sub["recall"] <= recall_threshold].copy()
        if sub.empty:
            continue

        zero_row = {
            "model": model_label,
            "checkpoint_nbr": 0,
            "tv": 0.0,
            "tv_stderr": 0.0,
            "kl": 0.0,
            "kl_stderr": 0.0,
        }
        sub = pd.concat([pd.DataFrame([zero_row]), sub], ignore_index=True).sort_values("checkpoint_nbr")

        color = color_map[model_label]
        marker = marker_style.get(model_label, 'o')
        y_col, yerr_col = ("tv", "tv_stderr") if mode == "TV" else ("kl", "kl_stderr")

        ax.plot(
            sub["checkpoint_nbr"],
            sub[y_col],
            label=labels[model_label],
            color=color,
            linestyle='-',
            marker=marker,
        )
        ax.fill_between(
            sub["checkpoint_nbr"],
            sub[y_col] - sub[yerr_col],
            sub[y_col] + sub[yerr_col],
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel(f"Train Step (until recall ≥ {recall_threshold})")
    ax.set_ylabel("Total Variation" if mode == "TV" else "KL Divergence")

    # Line style legend
    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle='-', linewidth=1.5, label="In-Weight"),
            Line2D([0], [0], color="black", linestyle='--', linewidth=1.5, label="In-Tool (worst)"),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.03, 0.62),
        ncol=1,
        frameon=False,
        framealpha=0.2,
        handletextpad=0.25,
        columnspacing=1.05,
        handlelength=1.1,
        title_fontsize=6.7,
    )
    ax.add_artist(linestyle_legend)

    # Family legend with model sizes
    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4,  label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-',linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-',linewidth=1.5, markersize=4, label="1.7B"),
    ]

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        save_name = "tv_vs_steps" if mode == "TV" else "kl_vs_steps"
    plt.savefig(os.path.join(save_path, f"{save_name}_{dataset_size}_{recall_threshold}_final.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{dataset_size}_{recall_threshold}_final.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----- Plot TotalVariation vs Training Steps (two plots side by side, different dataset_sizes)
def plot_tv_vs_training_steps_side_by_side(
    df,
    save_path,
    dataset_sizes=("500", "50k"),
    recall_threshold=0.9,
    mode="TV",
    save_name="tv_vs_steps_side_by_side"
):

    fig, axes = plt.subplots(1, 2, figsize=(3.8, 3.8/ 1.6), sharey=False)

    model_labels = [
        "Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"
    ]
    
    color_map = CMAP

    marker_style = {
        "Lam1B-weight": 'o', "Lam3B-weight": 'o', "Lam8B-weight": 'o',
        "Smol135M-weight": '^', "Smol360M-weight": '^', "Smol1.7B-weight": '^',
    }

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }
    
    # Global tool baseline: highest KL (or TV) across all dataset sizes
    tool_df_all = df[df["model"].str.contains("tool")]
    tool_baseline_y = tool_df_all["tv"].max() if mode == "TV" else tool_df_all["kl"].max()

    for ax, dataset_size in zip(axes, dataset_sizes):
        df_subset = df[df["dataset_size"] == dataset_size].copy()
        df_subset = df_subset.sort_values("checkpoint_nbr")

        # Plot tool baseline 
        ax.axhline(tool_baseline_y, color="black", linestyle="--", linewidth=1)

        for model_label in sorted(df_subset["model"].unique()):
            if "135" in model_label:
                continue
            if "tool" in model_label:
                continue
            sub = df_subset[df_subset["model"] == model_label]
            sub = sub[sub["recall"] <= recall_threshold].copy()
            if sub.empty:
                continue

            zero_row = {
                "model": model_label,
                "checkpoint_nbr": 0,
                "tv": 0.0,
                "tv_stderr": 0.0,
                "kl": 0.0,
                "kl_stderr": 0.0,
            }
            sub = pd.concat([pd.DataFrame([zero_row]), sub], ignore_index=True).sort_values("checkpoint_nbr")

            color = color_map[model_label]
            marker = marker_style.get(model_label, 'o')
            y_col, yerr_col = ("tv", "tv_stderr") if mode == "TV" else ("kl", "kl_stderr")

            ax.plot(
                sub["checkpoint_nbr"],
                sub[y_col],
                label=labels[model_label],
                color=color,
                linestyle='-',
                marker=marker,
            )
            ax.fill_between(
                sub["checkpoint_nbr"],
                sub[y_col] - sub[yerr_col],
                sub[y_col] + sub[yerr_col],
                color=color,
                alpha=0.15,
            )

        ax.grid(alpha=0.6)

    axes[0].set_ylabel("Total Variation" if mode == "TV" else "KL Divergence")
    fig.supxlabel(f"Training Step (until recall ≥ {recall_threshold})", y=0.11, fontsize=9) #fonsize

    # Legend: model sizes
    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="1.7B"),
    ]

    family_legend = fig.legend(
        handles=llama_handles + smol_handles,
        labels=["1B", "3B", "8B   ", "135M", "360M", "1.7B"],
        loc="upper center",
        bbox_to_anchor=(0.55, 1.04),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        handlelength=1.3,
        fontsize=7.6,
    )
    fig.add_artist(family_legend)
    fig.text(0.345, 1.025, "Llama Models", ha="center", fontsize=8)
    fig.text(0.715, 1.025, "SmolLM Models", ha="center", fontsize=8)

    # Legend: style
    linestyle_legend = axes[1].legend(
        handles=[
            Line2D([0], [0], color="black", linestyle='-', linewidth=1.25, label="In-Weight"),
            Line2D([0], [0], color="black", linestyle='--', linewidth=1.25, label="In-Tool (worst)"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.435, 0.57),
        ncol=1,
        frameon=False, #framealpha=0.2,
        handletextpad=0.25,
        columnspacing=1.,
        handlelength=1,
        fontsize=7.5
    )
    axes[1].add_artist(linestyle_legend)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{mode}_{recall_threshold}.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{mode}_{recall_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----------    Train Steps vs dataset size    ------------
def plot_train_steps_vs_dataset_size(df, save_path, recall_threshold=0.9, save_name="trainsteps_vs_facts"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5 / 1.3))

    size_map = {
        "0": 0, "500": 512, "1k": 1024, "5k": 2096,
        "10k": 8192, "50k": 50000, "all": 512
    }
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    model_labels = ["Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"]

    color_map = CMAP
    
    marker_style = {
        "Lam1B-weight": 'o', "Lam1B-tool": 'o',
        "Lam3B-weight": 'o', "Lam3B-tool": 'o',
        "Lam8B-weight": 'o', "Lam8B-tool": 'o',
        "Smol135M-weight": '^', "Smol135M-tool": '^',
        "Smol360M-weight": '^', "Smol360M-tool": '^',
        "Smol1.7B-weight": '^', "Smol1.7B-tool": '^',
    }

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }

    results = []

    for (model, dataset_size), group in df.groupby(["model", "dataset_size"]):
        if "tool" in model:
            continue  # Handle later
        group = group.sort_values("checkpoint_nbr")
        above_thresh = group[group["recall"] >= recall_threshold]
        if not above_thresh.empty:
            step = above_thresh.iloc[0]["checkpoint_nbr"]
        else:
            step = group.iloc[-1]["checkpoint_nbr"]
        results.append({
            "model": model,
            "dataset_size": dataset_size,
            "dataset_size_n": size_map.get(dataset_size, -1),
            "train_step": step
        })

    # Extend tool values to all sizes
        dataset_sizes_for_extension = ["500", "1k", "5k", "10k", "50k"]
    tool_rows = []
    tool_df = df[df["model"].str.contains("tool")]
    for model in tool_df["model"].unique():
        available_sizes = tool_df[tool_df["model"] == model]["dataset_size"].unique()
        if len(available_sizes) == 0:
            continue
        trained_size = available_sizes[0]  # e.g., "500", "1k", or "10k"
        trained_row = tool_df[(tool_df["model"] == model) & (tool_df["dataset_size"] == trained_size)]
        if trained_row.empty:
            continue
        train_step = trained_row["checkpoint_nbr"].max()  # be robust
        for size in dataset_sizes_for_extension:
            tool_rows.append({
                "model": model,
                "dataset_size": size,
                "dataset_size_n": size_map[size],
                "train_step": train_step,
            })

    full_df = pd.DataFrame(results + tool_rows)
    full_df = full_df.sort_values(["model", "dataset_size_n"])

    for idx, model_label in enumerate(sorted(full_df["model"].unique())):
        sub = full_df[full_df["model"] == model_label].sort_values("dataset_size_n")
        color = color_map[model_label]
        label = labels[model_label]
        x_vals = sub["dataset_size_n"].values
        y = sub["train_step"].values.astype(float)

        if "tool" in model_label:
            y = y + 0.05* (idx + 1)
            ax.plot(
                x_vals, y, color=color, label=label,
                linestyle="--", marker=marker_style.get(model_label, 'o'),
                linewidth=1.25, markersize=3.7
            )
        else:
            ax.plot(
                x_vals[1:], y[1:], color=color, label=label,
                linestyle="-", marker=marker_style.get(model_label, 'o'),
            )

    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1K", "5K", "10K", "50K"])
    ax.set_xlabel(f"Facts Memorized (Full Recall)")
    ax.set_ylabel("Training Steps")

    ax.set_yscale("log", base=10)
    tick_vals = [1, 1e1, 1e2, 1e3, 1e4]
    tick_labels = [1, r'$10$', r'$10^2$', r'$10^3$', r'$10^4$']
    ax.yaxis.set_major_locator(FixedLocator(tick_vals))
    ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))
        

    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle='-', linewidth=1.5, label="In-Weight"),
            Line2D([0], [0], color="black", linestyle='--', linewidth=1.5, label="In-Tool"),
        ],
        loc="upper left", ncol=2, frameon=False,
        handletextpad=0.25, columnspacing=1., handlelength=1.35,
    )
    ax.add_artist(linestyle_legend)

    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="1.7B"),
    ]
    family_legend = ax.legend(
        handles=llama_handles + smol_handles,
        labels=["1B", "3B", "8B     ", "135M", "360M", "1.7B"],
        loc="upper center",
        bbox_to_anchor=(0.495, 1.146),
        ncol=6, frameon=False, handletextpad=0.25,
        columnspacing=0.8, handlelength=1.3, fontsize=7.6
    )
    ax.add_artist(family_legend)
    ax.text(0.205, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.738, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{recall_threshold}.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{recall_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----------    Three metrics (subplots) vs training step    ------------
def plot_recall_hellaswag_tv(df, save_path, dataset_size="10k", recall_threshold=0.9, save_name="triple_plot_vs_steps"):
    fig, axes = plt.subplots(1, 3, figsize=(7, 7 / (3 * 1.2)), sharex=False)

    color_map = CMAP

    marker_style = {
        "Lam1B-weight": 'o', "Lam3B-weight": 'o', "Lam8B-weight": 'o',
        "Smol135M-weight": '^', "Smol360M-weight": '^', "Smol1.7B-weight": '^',
    }

    labels = {
        "Lam1B-weight": "1B-weight", "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight", "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight", "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight", "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight", "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight", "Smol1.7B-tool": "1.7B-tool",
    }

    # Get the globally worst in-tool model (by max TV)
    tool_df_all = df[df["model"].str.contains("tool")]
    if tool_df_all.empty:
        print("⚠️ No in-tool runs found.")
        return
    
    worst_tool_row = tool_df_all.loc[tool_df_all["tv"].idxmax()]
    worst_tool_model = worst_tool_row["model"]
    tool_df = tool_df_all[tool_df_all["model"] == worst_tool_model].copy()
    tool_df = tool_df.sort_values("checkpoint_nbr")

    # Filter in-weight runs for the selected dataset size
    df_subset = df[df["dataset_size"] == dataset_size].copy()
    df_subset = df_subset.sort_values("checkpoint_nbr")

    final_x = df_subset["checkpoint_nbr"].max()

    for i, metric in enumerate(["recall", "hellaswag_relative", "tv"]):
        ax = axes[i]
        yerr_col = {
            "recall": "recall_stderr",
            "hellaswag_relative": "hellaswag_relative_stderr",
            "tv": "tv_stderr"
        }[metric]
        y_label = {
            "recall": "Factual Recall",
            "hellaswag_relative": "HellaSwag Relative (%)",
            "tv": "Total Variation"
        }[metric]

        # === Plot in-tool actual trajectory ===
        ax.plot(
            tool_df["checkpoint_nbr"],
            tool_df[metric],
            color="black",
            linestyle="--",
            linewidth=1.25,
            marker='x',
            markersize=2.75,
            label="In-Tool"
        )

        # === Extend horizontal line from best in-tool point ===
        if metric == "hellaswag_relative":
            best_idx = tool_df[metric].idxmin()
        else:
            best_idx = tool_df[metric].idxmax()

        best_row = tool_df.loc[best_idx]
        best_x = best_row["checkpoint_nbr"]
        best_y = best_row[metric]
        #final_x = tool_df["checkpoint_nbr"].max()

        ax.hlines(
            y=best_y,
            xmin=best_x,
            xmax=final_x,
            color="black",
            linestyle='--',
            linewidth=1.1
        )

        # === Plot in-weight models ===
        for model_label in sorted(df_subset["model"].unique()):
            if "tool" in model_label:
                continue
            sub = df_subset[df_subset["model"] == model_label].copy()
            sub = sub[sub["recall"] <= recall_threshold]
            if sub.empty:
                continue

            # Add zero-step row
            zero_row = {
                "model": model_label,
                "checkpoint_nbr": 0,
                metric: 100.0 if metric == "hellaswag_relative" else 0.0,
                yerr_col: 0.0,
            }
            sub = pd.concat([pd.DataFrame([zero_row]), sub], ignore_index=True)
            sub = sub.sort_values("checkpoint_nbr")

            ax.plot(
                sub["checkpoint_nbr"],
                sub[metric],
                color=color_map[model_label],
                label=labels[model_label],
                marker=marker_style.get(model_label, 'o'),
                markersize=3,
                linestyle='-'
            )
            upper = (
                (sub[metric] + sub[yerr_col]).clip(upper=100)
                if metric == "hellaswag_relative"
                else sub[metric] + sub[yerr_col]
            )
            ax.fill_between(
                sub["checkpoint_nbr"],
                sub[metric] - sub[yerr_col],
                upper,
                color=color_map[model_label],
                alpha=0.15
            )

        ax.set_xlabel("Train Step")
        ax.set_ylabel(y_label)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Legend: Llama and Smol
    llama_handles = [
        Line2D([0], [0], color=color_map["Lam1B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="1B"),
        Line2D([0], [0], color=color_map["Lam3B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="3B"),
        Line2D([0], [0], color=color_map["Lam8B-weight"], marker='o', linestyle='-', linewidth=1.5, markersize=4, label="8B"),
    ]
    smol_handles = [
        Line2D([0], [0], color=color_map["Smol135M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="135M"),
        Line2D([0], [0], color=color_map["Smol360M-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="360M"),
        Line2D([0], [0], color=color_map["Smol1.7B-weight"], marker='^', linestyle='-', linewidth=1.5, markersize=4, label="1.7B"),
    ]

    fig.legend(
        handles=llama_handles + smol_handles,
        labels=["1B", "3B", "8B        ", "135M", "360M", "1.7B"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.8,
        handlelength=1.45,
        fontsize=7.6
    )
    fig.text(0.368, 1.065, "Llama Models", ha="center", fontsize=8)
    fig.text(0.61, 1.065, "SmolLM Models", ha="center", fontsize=8)

    if dataset_size == "500":
         # Line style legend
        linestyle_legend = axes[1].legend(
            handles=[
                Line2D([0], [0], color="black", linestyle='-', linewidth=1.2, label="In-Weight"),
                Line2D([0], [0], color="black", linestyle='--', linewidth=1.2, marker='x', label="In-Tool (worst)"),
            ],
            bbox_to_anchor=(0.45, 0.3), #(0.45, 0.3)
            ncol=1,
            frameon=False,
            handletextpad=0.25,
            columnspacing=1.05,
            handlelength=1.35,
            fontsize=7.3
        )
        axes[1].add_artist(linestyle_legend)

    else:
         # Line style legend
        linestyle_legend = axes[0].legend(
            handles=[
                Line2D([0], [0], color="black", linestyle='-', linewidth=1.2, label="In-Weight"),
                Line2D([0], [0], color="black", linestyle='--', linewidth=1.2, marker='x', label="In-Tool (worst)"),
            ],
            loc="lower right", #bbox_to_anchor=(0.45, 0.3), #(0.45, 0.3)
            ncol=1,
            frameon=False,
            handletextpad=0.25,
            columnspacing=1.05,
            handlelength=1.35,
            fontsize=7.3
        )
        axes[0].add_artist(linestyle_legend)


    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{dataset_size}_{recall_threshold}.pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(os.path.join(save_path, f"{save_name}_{dataset_size}_{recall_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----------    Compute table with avg training steps for each run on    ------------
def aggregate_training_steps_for_latex(df, save_path, save_name="avg_training_steps.csv"):
    df_filtered = df[df["recall"] > 0.95].copy()
    df_filtered["family"] = df_filtered["model"].apply(lambda x: "in-tool" if "tool" in x else "in-weight")
    df_filtered["model_size"] = df_filtered["model"].str.extract(r'(Smol135M|Smol360M|Smol1.7B|Lam1B|Lam3B|Lam8B)')

    # Separate in-weight and in-tool
    in_weight_df = df_filtered[df_filtered["family"] == "in-weight"]
    in_tool_df = df_filtered[df_filtered["family"] == "in-tool"]

    # --- Handle in-weight models ---
    grouped_weight = in_weight_df.groupby(["model", "dataset_size"])
    min_ckpt_weight = grouped_weight["checkpoint_nbr"].min().reset_index()
    min_ckpt_weight = pd.merge(
        min_ckpt_weight,
        in_weight_df[["model", "dataset_size", "family", "model_size"]].drop_duplicates(),
        on=["model", "dataset_size"],
        how="left"
    )

    # --- Handle in-tool models ---
    # Step 1: Find the dataset size where each tool model was trained
    tool_ckpt = (
        in_tool_df.groupby("model")
        .apply(lambda g: g.loc[g["checkpoint_nbr"].idxmin()])
        .reset_index(drop=True)
    )

    # Step 2: Duplicate this across all dataset sizes
    all_dataset_sizes = df["dataset_size"].unique()
    expanded_tool_rows = []
    for _, row in tool_ckpt.iterrows():
        for ds in all_dataset_sizes:
            expanded_tool_rows.append({
                "model": row["model"],
                "dataset_size": ds,
                "checkpoint_nbr": row["checkpoint_nbr"],
                "family": row["family"],
                "model_size": row["model_size"]
            })
    expanded_tool_df = pd.DataFrame(expanded_tool_rows)

    # --- Combine and compute averages ---
    merged = pd.concat([min_ckpt_weight, expanded_tool_df], ignore_index=True)
    avg_steps = merged.groupby(["family", "dataset_size"])["checkpoint_nbr"].mean().reset_index()
    avg_steps = avg_steps.pivot(index="dataset_size", columns="family", values="checkpoint_nbr").reset_index()

    # Save 
    avg_steps.to_csv(f"{save_path}/{save_name}", index=False)
    print("Table saved as avg_training_steps.csv and avg_training_steps.tex")
    return avg_steps


class PlotCLI:
    def __init__(self):
        self.RESULT_PATH = Path(__file__).parents[1] / "large_scale_results.csv"
        self.FIGURE_PATH = Path(__file__).parents[1] / "Plots"

    def plot_all(self, csv_file=None, recall_threshold=0.95):
        """
        Run all plots in sequence.
        
        Usage:
            python plot_cli.py plot_all --recall_threshold=0.95
            python plot_cli.py plot_all --csv_file="path/to/results.csv"
        """
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")

        # -----  Hellaswag vs dataset_size 
        plot_hellaswag_vs_datasetsize_absolute(full_df, self.FIGURE_PATH, acc_threshold=recall_threshold, save_name="hellaswag_vs_facts_absolute")
        plot_hellaswag_vs_datasetsize_relative(full_df, self.FIGURE_PATH, acc_threshold=recall_threshold, save_name="hellaswag_vs_facts_relative")

        # -----  Total Variation vs dataset_size 
        plot_final_tv_vs_dataset_size(full_df, self.FIGURE_PATH, recall_threshold=recall_threshold, save_name="final_kl_vs_facts")
        plot_final_tv_vs_dataset_size_withtool(full_df, self.FIGURE_PATH, recall_threshold=recall_threshold, save_name="kl_vs_facts_withtool")

        # -----  Total Variation vs train_steps (mode="TV" or "KL" for Total Variation or KL divergence resp.)
        plot_tv_vs_training_steps(full_df, self.FIGURE_PATH, dataset_size="500", recall_threshold=recall_threshold, mode="TV")

        # -----  Total Variation vs train_steps (2 subplots, size by side)
        plot_tv_vs_training_steps_side_by_side(full_df, self.FIGURE_PATH, dataset_sizes=("500", "50k"), recall_threshold=recall_threshold, mode="TV", save_name="tv_vs_steps_side_by_side_no135")

        # -----  Train Steps vs dataset_size
        plot_train_steps_vs_dataset_size(full_df, self.FIGURE_PATH, recall_threshold=recall_threshold, save_name="trainstep_vs_facts")

        # -----  Recall & Hellaswag & Total Variation vs dataset_size (three subplots)
        plot_recall_hellaswag_tv(full_df, self.FIGURE_PATH, dataset_size="500", recall_threshold=recall_threshold, save_name="triple_plot_vs_steps")

        # -----  Compute training steps average by each model, for each dataset:
        aggregate_training_steps_for_latex(full_df, self.FIGURE_PATH, save_name="avg_training_steps.csv")


    def plot_hellaswag_absolute(self, csv_file=None, acc_threshold=0.95):
        """
        Plot Hellaswag accuracy (absolute) vs dataset size.
        Usage:
            python plot_cli.py plot_hellaswag_absolute --acc_threshold=0.9
        """
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_hellaswag_vs_datasetsize_absolute(full_df, self.FIGURE_PATH, acc_threshold, save_name="hellaswag_vs_facts_absolute")

    def plot_hellaswag_relative(self, csv_file=None, acc_threshold=0.95):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_hellaswag_vs_datasetsize_relative(full_df, self.FIGURE_PATH, acc_threshold, save_name="hellaswag_vs_facts_relative")

    def plot_tv_vs_dataset_size(self, csv_file=None, recall_threshold=0.95):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_final_tv_vs_dataset_size(full_df, self.FIGURE_PATH, recall_threshold, save_name="final_kl_vs_facts")

    def plot_tv_vs_dataset_size_withtool(self, csv_file=None, recall_threshold=0.95):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_final_tv_vs_dataset_size_withtool(full_df, self.FIGURE_PATH, recall_threshold, save_name="kl_vs_facts_withtool")

    def plot_tv_vs_train_steps(self, csv_file=None, recall_threshold=0.95, dataset_size="500", mode="TV"):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_tv_vs_training_steps(full_df, self.FIGURE_PATH, dataset_size=dataset_size, recall_threshold=recall_threshold, mode=mode)

    def plot_tv_vs_train_steps_side_by_side(self, csv_file=None, recall_threshold=0.95, dataset_sizes=("500", "50k"), mode="TV"):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_tv_vs_training_steps_side_by_side(full_df, self.FIGURE_PATH, dataset_sizes=dataset_sizes, recall_threshold=recall_threshold, mode=mode, save_name="tv_vs_steps_side_by_side_no135")

    def plot_train_steps_vs_dataset_size(self, csv_file=None, recall_threshold=1.0):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_train_steps_vs_dataset_size(full_df, self.FIGURE_PATH, recall_threshold=recall_threshold, save_name="trainstep_vs_facts")

    def plot_triple_vs_train_steps(self, csv_file=None, recall_threshold=1.0):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        plot_recall_hellaswag_tv(full_df, self.FIGURE_PATH, dataset_size="500", recall_threshold=recall_threshold, save_name="triple_plot_vs_steps")

    def aggregate_training_steps(self, csv_file=None):
        full_df = pd.read_csv(csv_file or f"{self.RESULT_PATH}/large_scale_results.csv")
        aggregate_training_steps_for_latex(full_df, self.FIGURE_PATH, save_name="avg_training_steps.csv")

if __name__ == "__main__":
    fire.Fire(PlotCLI)
    # -------------------------------
    # Example usage:
    #
    # 1. Run all plots with default recall threshold (0.95):
    #    python -m apps.memory.plots.large_scale_analysis plot_all
    #
    # 2. Run all plots with custom recall threshold:
    #    python -m apps.memory.plots.large_scale_analysis plot_all --recall_threshold=0.9
    #
    # 3. Plot only Hellaswag absolute accuracy vs dataset size:
    #    python -m apps.memory.plots.large_scale_analysis plot_hellaswag_absolute --acc_threshold=0.9
    #
    # 4. Plot TV vs training steps for dataset_size=50k and mode=KL:
    #    python -m apps.memory.plots.large_scale_analysis plot_tv_vs_train_steps --dataset_size=50k --mode=KL
    #
    # 5. Aggregate training steps for LaTeX output:
    #    python -m apps.memory.plots.large_scale_analysis aggregate_training_steps
    #
    # -------------------------------

