"""
TODO: clean code
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from matplotlib import rcParams
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

import matplotlib.cm as cm

# Plotting format
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

rcParams.update(
    {
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
    }
)


# ===========================================================================
# ==========    Code to Load and Create DataFrame of Results    =============
# ===========================================================================


# --------  Code to collect eval results for hellaswag, recall and kl
def collect_all_recall_results(base_path: Path):
    all_records = []
    for subdir in base_path.glob("Results_*_final"):  # change this to Resulst
        recall_dir = subdir / "Recall"
        if recall_dir.exists() and recall_dir.is_dir():
            print(f"[Info] Found recall dir: {recall_dir}")
            records = collect_recall_results(recall_dir)
            all_records.extend(records)
        else:
            print(f"[Skip] No recall directory in {subdir}")
    return all_records


def collect_recall_results(base_path: Path):
    records_by_run = defaultdict(list)
    for json_file in base_path.rglob("checkpoints_recall_results.json"):
        try:
            with open(json_file) as f:
                results = json.load(f)
            run_info = json_file.parent.name
            for ckpt_name, ckpt_data in results.items():
                if not isinstance(ckpt_data, dict):
                    continue
                checkpoint = int(ckpt_name.replace("checkpoint-", ""))
                records_by_run[run_info].append(
                    {
                        "type": "recall",
                        "run": run_info,
                        "checkpoint": checkpoint,
                        "recall_accuracy_test": ckpt_data.get("accuracy_test"),
                        "recall_stderr_test": ckpt_data.get("stderr_test"),
                        "recall_accuracy_train": ckpt_data.get("accuracy_train"),
                        "recall_stderr_train": ckpt_data.get("stderr_train"),
                        "recall_num_eval_test": ckpt_data.get("num_eval_test"),
                        "recall_num_eval_train": ckpt_data.get("num_eval_train"),
                        "path": str(json_file),
                    }
                )
        except Exception as e:
            print(f"Failed to parse Recall file {json_file}: {e}")
    # Add epoch numbers
    final_records = []
    for run, recs in records_by_run.items():
        sorted_recs = sorted(recs, key=lambda x: x["checkpoint"])
        for epoch_idx, rec in enumerate(sorted_recs, start=1):
            rec["epoch"] = epoch_idx
            final_records.append(rec)
    return final_records


def collect_all_Hellaswag_results(base_path: Path):
    all_records = []
    for subdir in base_path.glob("Results_*_final"):
        hella_dir = subdir / "Hellaswag"
        if hella_dir.exists() and hella_dir.is_dir():
            print(f"[Info] Found Hellaswag dir: {hella_dir}")
            records = collect_hellaswag_results(hella_dir)
            all_records.extend(records)
        else:
            print(f"[Skip] No Hellaswag directory in {subdir}")
    return all_records


def collect_hellaswag_results(base_path: Path):
    records_by_run = defaultdict(list)
    run_dir = base_path / "runs"
    for result_file in run_dir.rglob("results_*.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)
            hellaswag_result = result["results"]["hellaswag"]
            acc = hellaswag_result.get("acc,none")
            acc_stderr = hellaswag_result.get("acc_stderr,none")

            run_info = result_file.parts[result_file.parts.index("runs") + 1]
            checkpoint_match = re.search(r"checkpoint-(\d+)", str(result_file))
            checkpoint_step = int(checkpoint_match.group(1)) if checkpoint_match else -1

            records_by_run[run_info].append(
                {
                    "type": "hellaswag",
                    "run": run_info,
                    "checkpoint": checkpoint_step,
                    "hellaswag_accuracy": acc,
                    "hellaswag_stderr": acc_stderr,
                    "path": str(result_file),
                }
            )
        except Exception as e:
            print(f"Failed to load Hellaswag result {result_file}: {e}")
    # Add epoch numbers
    final_records = []
    for run, recs in records_by_run.items():
        sorted_recs = sorted(recs, key=lambda x: x["checkpoint"])
        for epoch_idx, rec in enumerate(sorted_recs, start=1):
            rec["epoch"] = epoch_idx
            final_records.append(rec)
    return final_records


def collect_all_KL_TV_results(base_path: Path):
    all_records = []
    for subdir in base_path.glob("Results_*_final"):
        kl_dir = subdir / "KL"
        if kl_dir.exists() and kl_dir.is_dir():
            print(f"[Info] Found kl dir: {kl_dir}")
            records = collect_kl_tv_results(kl_dir)
            all_records.extend(records)
        else:
            print(f"[Skip] No recall directory in {subdir}")
    return all_records


def collect_kl_tv_results(base_path: Path):
    records_by_run = defaultdict(list)
    for json_file in base_path.rglob("checkpoint_kl_eval_results.json"):
        try:
            with open(json_file) as f:
                results = json.load(f)
            run_info = json_file.parent.name
            for ckpt_name, ckpt_data in results.items():
                if not isinstance(ckpt_data, dict):
                    continue
                checkpoint = int(ckpt_name.replace("checkpoint-", ""))
                records_by_run[run_info].append(
                    {
                        "type": "kl",
                        "run": run_info,
                        "checkpoint": checkpoint,
                        "kl_mean": ckpt_data.get("kl_mean"),
                        "kl_stderr": ckpt_data.get("kl_stderr"),
                        "tv_mean": ckpt_data.get("tv_mean"),
                        "tv_stderr": ckpt_data.get("tv_stderr"),
                        "num_prompts_for_eval": ckpt_data.get("num_eval_kl"),
                        "path": str(json_file),
                    }
                )
        except Exception as e:
            print(f"Failed to parse Recall file {json_file}: {e}")
    # Add epoch numbers
    final_records = []
    for run, recs in records_by_run.items():
        sorted_recs = sorted(recs, key=lambda x: x["checkpoint"])
        for epoch_idx, rec in enumerate(sorted_recs, start=1):
            rec["epoch"] = epoch_idx
            final_records.append(rec)

    return final_records


# =============================================================
# ==========    Code to Fetch Specific Results    =============
# =============================================================


def get_recall_hellaswag_tv_run_data(recall_df, tv_df, hellaswag_df, eval_results_path, acc_threshold=1):
    # Fetch base model hellaswag data
    base_models_paths = {
        "Lam1B": f"{eval_results_path}/Results_llama_final/Hellaswag/base_models/Llama-3.2-1B-Instruct/results_2025-05-06T23-30-06.017723.json",
        "Lam3B": f"{eval_results_path}/Results_llama_final/Hellaswag/base_models/Llama-3.2-3B-Instruct/results_2025-05-07T00-07-24.007108.json",
        "Lam8B": f"{eval_results_path}/Results_llama_final/Hellaswag/base_models/Llama-3.1-8B-Instruct/meta-llama__Llama-3.1-8B-Instruct/results_2025-05-19T10-57-37.932935.json",
        "Smol135M": f"{eval_results_path}/Results_smol_final/Hellaswag/base_models/SmolLM-135M-Instruct/HuggingFaceTB__SmolLM-135M-Instruct/results_2025-06-28T12-52-52.265790.json",
        "Smol360M": f"{eval_results_path}/Results_smol_final/Hellaswag/base_models/SmolLM-360M_Instruct/HuggingFaceTB__SmolLM-360M-Instruct/results_2025-06-28T05-07-18.025099.json",
        "Smol1.7B": f"{eval_results_path}/Results_smol_final/Hellaswag/base_models/SmolLM-1.7B-Instruct/HuggingFaceTB__SmolLM-1.7B-Instruct/results_2025-06-28T05-09-42.516537.json",
    }

    base_scores = {}
    for model_id, path in base_models_paths.items():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            acc = data["results"]["hellaswag"]["acc,none"]
            stderr = data["results"]["hellaswag"]["acc_stderr,none"]
            base_scores[model_id] = (acc, stderr)
        except Exception as e:
            print(f"⚠️ Failed to load base model score for {model_id}: {e}")
            base_scores[model_id] = (None, None)

    # Define runs
    run_names = {
        # Llama models
        "Lam1B-weight": {
            "500": "sft_Lam1B_facts=500-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight",
            "1k": "sft_Lam1B_facts=1000-epochs=20-batch=4-gradAcc=32-LR=5e-5-loraR=0-loraA=0-weight",
            "5k": "sft_Lam1B_facts=5000-epochs=25-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight",
            "10k": "sft_Lam1B_facts=10000-epochs=20-batch=4-gradAcc=32-LR=5e-5-loraR=0-loraA=0-weight",
            "50k": "sft_Lam1B_facts=50000-epochs=20-batch=4-gradAcc=32-LR=5e-5-loraR=0-loraA=0-weight",
        },
        "Lam3B-weight": {
            "500": "sft_Lam3B_facts=500-epochs=50-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight",
            "1k": "sft_Lam3B_facts=1000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight",
            "5k": "sft_Lam3B_facts=5000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight",
            "10k": "sft_Lam3B_facts=10000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight",
            "50k": "sft_Lam3B_facts=50000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight",
        },
        "Lam8B-weight": {
            "500": "sft_Lam8B_facts=500-epochs=50-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight",
            "1k": "sft_Lam8B_facts=1000-epochs=80-batch=4-gradAcc=64-LR=2e-5-loraR=0-loraA=0-weight",
            "5k": "sft_Lam8B_facts=5000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight",
            "10k": "sft_Lam8B_facts=10000-epochs=80-batch=4-gradAcc=64-LR=2e-5-loraR=0-loraA=0-weight",
            "50k": "sft_Lam8B_facts=50000-epochs=80-batch=4-gradAcc=64-LR=2e-5-loraR=0-loraA=0-weight",
        },
        "Lam1B-tool": {"500": "sft_Lam1B_facts=500-epochs=15-batch=4-gradAcc=32-LR=5e-5-loraR=0-loraA=0-tool"},
        "Lam3B-tool": {"500": "sft_Lam3B_facts=500-epochs=15-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-tool"},
        "Lam8B-tool": {"500": "sft_Lam8B_facts=500-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-tool"},
        # SmolLM models
        "Smol135M-weight": {
            "500": "sft_Smol135M_facts=500-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "1k": "sft_Smol135M_facts=1000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "5k": "sft_Smol135M_facts=5000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "10k": "sft_Smol135M_facts=10000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "50k": "sft_Smol135M_facts=50000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
        },
        "Smol360M-weight": {
            "500": "sft_Smol360M_facts=500-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "1k": "sft_Smol360M_facts=1000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "5k": "sft_Smol360M_facts=5000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "10k": "sft_Smol360M_facts=10000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
            "50k": "sft_Smol360M_facts=50000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight",
        },
        "Smol1.7B-weight": {
            "500": "sft_Smol1.7B_facts=500-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight",
            "1k": "sft_Smol1.7B_facts=1000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight",
            "5k": "sft_Smol1.7B_facts=5000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight",
            "10k": "sft_Smol1.7B_facts=10000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight",
            "50k": "sft_Smol1.7B_facts=50000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight",
        },
        "Smol135M-tool": {"all": "sft_Smol135M_facts=1000-epochs=15-batch=64-gradAcc=2-LR=1e-4-loraR=0-loraA=0-tool"},
        "Smol360M-tool": {"all": "sft_Smol360M_facts=1000-epochs=15-batch=64-gradAcc=1-LR=1e-4-loraR=0-loraA=0-tool"},
        "Smol1.7B-tool": {"all": "sft_Smol1.7B_facts=1000-epochs=15-batch=64-gradAcc=2-LR=5e-5-loraR=0-loraA=0-tool"},
    }

    data = []
    for model_label, data_sizes in run_names.items():
        if "135M" in model_label:
            model_id = "Smol135M"
        elif "360M" in model_label:
            model_id = "Smol360M"
        elif "1.7B" in model_label:
            model_id = "Smol1.7B"
        elif "1B" in model_label:
            model_id = "Lam1B"
        elif "3B" in model_label:
            model_id = "Lam3B"
        elif "8B" in model_label:
            model_id = "Lam8B"
        else:
            print(f"\n --- Problem: model_label not well formed: {model_label}")

        # Get corresponding base_model hellaswag score
        base_score, base_stderr = base_scores.get(model_id, (None, None))
        if base_score is None or base_stderr is None:
            print(f"⚠️ Skipping {model_label}: base score missing.")
            continue

        for data_size_label, run in data_sizes.items():
            # Determine checkpoints until threshold accuracy
            if "tool" in run:
                passing = recall_df[
                    (recall_df["run"] == run) & (recall_df["recall_accuracy_test"] >= acc_threshold * 0.99)
                ]
            else:
                passing = recall_df[
                    (recall_df["run"] == run) & (recall_df["recall_accuracy_train"] >= acc_threshold * 0.99)
                ]

            # Filter out checkpoints beyond
            if passing.empty:
                # Use all checkpoints for the run
                rec_range = recall_df[recall_df["run"] == run].sort_values("checkpoint")
                print(f"⏩ No recall ≥ {acc_threshold} for {run}, using all {len(rec_range)} checkpoints.")
            else:
                first_ckpt = passing.sort_values("checkpoint").iloc[0]["checkpoint"]
                rec_range = recall_df[(recall_df["run"] == run) & (recall_df["checkpoint"] <= first_ckpt)].sort_values(
                    "checkpoint"
                )

            # Order checkpoints, collect statistics for each
            rec_range = rec_range.sort_values("checkpoint")
            for _, rec in rec_range.iterrows():
                # Checkpoint number
                ckpt = rec["checkpoint"]

                # Fetch KL and TV data
                tv_row = tv_df[(tv_df["run"] == run) & (tv_df["checkpoint"] == ckpt)]
                if not tv_row.empty:
                    tv = tv_row.iloc[0]
                    kl_mean, kl_stderr = tv["kl_mean"], tv["kl_stderr"]
                    tv_mean, tv_stderr = tv["tv_mean"], tv["tv_stderr"]
                else:
                    kl_mean = kl_stderr = tv_mean = tv_stderr = None

                # Fetch Hellaswag scores (fintuned checkpoints)
                hs_row = hellaswag_df[(hellaswag_df["run"] == run) & (hellaswag_df["checkpoint"] == ckpt)]
                if hs_row.empty:
                    continue
                hs = hs_row.iloc[0]
                fine_score = hs.get("hellaswag_accuracy")
                fine_stderr = hs.get("hellaswag_stderr")
                if fine_score is None or fine_score <= 0:
                    continue

                # Compute Hellaswag score relative to base_model
                epsilon = 1e-8
                relative = 100 * fine_score / (base_score + epsilon)
                relative_stderr = (
                    relative
                    * ((fine_stderr / (fine_score + epsilon)) ** 2 + (base_stderr / (base_score + epsilon)) ** 2) ** 0.5
                )

                data.append(
                    {
                        "run": run,
                        "model": model_label,
                        "dataset_size": data_size_label,
                        "checkpoint_nbr": ckpt,
                        "recall": rec["recall_accuracy_train"],
                        "recall_stderr": rec["recall_stderr_train"],
                        "hellaswag_absolute": fine_score,
                        "hellaswag_absolute_stderr": fine_stderr,
                        "hellaswag_relative": relative,
                        "hellaswag_relative_stderr": relative_stderr,
                        "tv": tv_mean,
                        "tv_stderr": tv_stderr,
                        "kl": kl_mean,
                        "kl_stderr": kl_stderr,
                    }
                )

                if not any((d["model"] == model_label and d["dataset_size"] == "0") for d in data):
                    data.append(
                        {
                            "run": f"{model_label}_base",
                            "model": model_label,
                            "dataset_size": "0",
                            "checkpoint_nbr": 0,
                            "recall": 0.0,
                            "recall_stderr": 0.0,
                            "hellaswag_absolute": base_score,
                            "hellaswag_absolute_stderr": base_stderr,
                            "hellaswag_relative": 100.0,
                            "hellaswag_relative_stderr": 0.0,
                            "tv": 0.0,
                            "tv_stderr": 0.0,
                            "kl": 0.0,
                            "kl_stderr": 0.0,
                        }
                    )
    return pd.DataFrame(data)


# =================================================
# ==========    Plotting Functions    =============
# =================================================

# ----------    HellaSwag Plots    ------------


# ---- Plot Hellaswag_performace (absolute) vs Dataset_size
def plot_hellaswag_vs_datasetsize_absolute(df, save_path, acc_threshold, save_name="hellaswag_vs_facts_absolute"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5 / 1.3))  # ICLR-style dimensions

    size_map = {"0": 0, "500": 500, "1k": 1000, "5k": 5000, "10k": 10000, "50k": 50000, "all": 500}
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    model_labels = ["Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"]

    # Use Turbo or Viridis — Turbo matches better visually
    cmap = cm.get_cmap("viridis", len(model_labels))
    # Truncate to avoid the lightest ~10% of the colormap
    start, end = 0.04, 0.84
    color_map = {}

    for i, model in enumerate(model_labels):
        frac = 1.0 - i / (len(model_labels) - 1)  # reverse order
        truncated_frac = start + frac * (end - start)
        color = cmap(truncated_frac)
        hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
        for suffix in ["-weight", "-tool"]:
            color_map[f"{model}{suffix}"] = hex_color

        marker_style = {
            "Lam1B-weight": "o",
            "Lam3B-weight": "o",
            "Lam8B-weight": "o",
            "Smol135M-weight": "^",
            "Smol360M-weight": "^",
            "Smol1.7B-weight": "^",
        }

    labels = {
        "Lam1B-weight": "1B-weight",
        "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight",
        "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight",
        "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight",
        "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight",
        "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight",
        "Smol1.7B-tool": "1.7B-tool",
    }

    # Plotting x-axis scale:
    size_map = {
        "0": 350,  # fake-log position for 0
        "500": 512,
        "1k": 1024,
        "5k": 2096,
        "10k": 8192,
        "50k": 50000,
        "all": 512,  # map to same as 500 for consistency
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
            # x_vals = sub["dataset_size_n"].values
            y_mean = sub["hellaswag_absolute"].mean()
            y_std = sub["hellaswag_absolute_stderr"].mean()
            ax.plot(x_vals, [y_mean] * len(x_vals), linestyle="--", color=color, label=label, linewidth=1)
            ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)
        else:
            x = sub["dataset_size_n"].values
            y = sub["hellaswag_absolute"].values
            yerr = sub["hellaswag_absolute_stderr"].values
            ax.plot(x, y, color=color, label=label, marker=marker_style.get(model_label, "o"))  # linewidth=1.75
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    # #ax.set_xscale("log")
    # ax.set_xticks([0, 500, 1000, 5000, 10000, 50000])
    # ax.set_xticklabels(["0", "500", "1k", "5k", "10k", "50k"])
    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1k", "5k", "10k", "50k"])
    ax.set_xlabel(f"Facts Memorized (≥{acc_threshold} recall)")
    ax.set_ylabel("HellaSwag Accuracy")

    # Legend for line style (in-weight vs in-tool)
    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="weight"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="tool"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.02),
        ncol=2,
        frameon=False,  # False
        handletextpad=0.25,
        columnspacing=1.0,
        handlelength=1.35,
    )
    ax.add_artist(linestyle_legend)

    # Family legend with model sizes
    llama_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Lam1B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam3B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="3B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam8B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="8B",
        ),
    ]
    smol_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Smol135M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="135M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol360M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="360M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol1.7B-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1.7B",
        ),
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
        fontsize=7.6,
    )

    ax.add_artist(family_legend)

    # Add manual group titles for families
    ax.text(0.205, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.738, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{acc_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ---- Plot Hellaswag_performace (absolute) vs Dataset_size
def plot_hellaswag_vs_datasetsize_relative(df, save_path, acc_threshold, save_name="hellaswag_vs_facts_relative"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5 / 1.3))  # ICLR-style dimensions

    size_map = {"0": 0, "500": 500, "1k": 1000, "5k": 5000, "10k": 10000, "50k": 50000, "all": 500}
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    color_map = {
        # Lam
        "Lam1B-weight": "#A6D5C8",
        "Lam3B-weight": "#5EAAA8",
        "Lam8B-weight": "#1F4E5F",
        "Lam1B-tool": "#A6D5C8",
        "Lam3B-tool": "#5EAAA8",
        "Lam8B-tool": "#1F4E5F",
        # Smol
        "Smol135M-weight": "#D9B3DD",
        "Smol360M-weight": "#C48CC9",
        "Smol1.7B-weight": "#803A94",
        "Smol135M-tool": "#D9B3DD",
        "Smol360M-tool": "#C48CC9",
        "Smol1.7B-tool": "#803A94",
    }

    model_labels = ["Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"]

    # Use Turbo or Viridis — Turbo matches better visually
    cmap = cm.get_cmap("viridis", len(model_labels))
    # Truncate to avoid the lightest ~10% of the colormap
    start, end = 0.04, 0.84
    color_map = {}

    for i, model in enumerate(model_labels):
        frac = 1.0 - i / (len(model_labels) - 1)  # reverse order
        truncated_frac = start + frac * (end - start)
        color = cmap(truncated_frac)
        hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
        for suffix in ["-weight", "-tool"]:
            color_map[f"{model}{suffix}"] = hex_color

        marker_style = {
            "Lam1B-weight": "o",
            "Lam3B-weight": "o",
            "Lam8B-weight": "o",
            "Smol135M-weight": "^",
            "Smol360M-weight": "^",
            "Smol1.7B-weight": "^",
        }

    labels = {
        "Lam1B-weight": "1B-weight",
        "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight",
        "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight",
        "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight",
        "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight",
        "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight",
        "Smol1.7B-tool": "1.7B-tool",
    }

    # Plotting x-axis scale:
    size_map = {
        "0": 350,  # fake-log position for 0
        "500": 512,
        "1k": 1024,
        "5k": 2096,
        "10k": 8192,
        "50k": 50000,
        "all": 512,  # map to same as 500 for consistency
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
            if not tool_baseline_plotted:  # Only plot the first since they are about the same
                # Horizontal dashed line with shaded area
                x_vals = [350, 500, 1000, 5000, 10000, 50000]
                # x_vals = sub["dataset_size_n"].values
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
            ax.plot(x, y, color=color, label=label, marker=marker_style.get(model_label, "o"))  # linewidth=1.75
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    # #ax.set_xscale("log")
    # ax.set_xticks([0, 500, 1000, 5000, 10000, 50000])
    # ax.set_xticklabels(["0", "500", "1k", "5k", "10k", "50k"])
    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1k", "5k", "10k", "50k"])
    ax.set_xlabel(f"Facts Memorized (≥{acc_threshold} recall)")
    ax.set_ylabel("HellaSwag Performace", labelpad=13)
    ax.text(
        -0.12,
        0.5,  # x = left of axis, y = centered vertically
        "(% relative to base model)",
        fontsize=8.5,
        rotation=90,
        va="center",
        ha="right",
        transform=ax.transAxes,
    )

    # Legend for line style (in-weight vs in-tool)
    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="weight"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="tool baseline"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.01),
        ncol=2,
        frameon=False,  # False
        handletextpad=0.25,
        columnspacing=1.0,
        handlelength=1.38,
    )
    ax.add_artist(linestyle_legend)

    # Family legend with model sizes
    llama_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Lam1B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam3B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="3B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam8B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="8B",
        ),
    ]
    smol_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Smol135M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="135M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol360M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="360M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol1.7B-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1.7B",
        ),
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
        fontsize=7.6,
    )
    ax.add_artist(family_legend)

    # Add manual group titles for families
    ax.text(0.209, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.746, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{acc_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----------    Total Variation Plots    ------------


# ----- Plot TotalVariation vs Dataset_Size
def plot_final_tv_vs_dataset_size(df, save_path, recall_threshold=0.9, save_name="final_tv_vs_facts"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5 / 1.3))  # ICLR-style

    size_map = {"0": 0, "500": 512, "1k": 1024, "5k": 2096, "10k": 8192, "50k": 50000, "all": 512}
    df["dataset_size_n"] = df["dataset_size"].map(size_map)

    model_labels = ["Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"]
    cmap = cm.get_cmap("viridis", len(model_labels))
    start, end = 0.04, 0.84
    color_map = {}
    for i, model in enumerate(model_labels):
        frac = 1.0 - i / (len(model_labels) - 1)
        truncated_frac = start + frac * (end - start)
        color = cmap(truncated_frac)
        hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
        for suffix in ["-weight", "-tool"]:
            color_map[f"{model}{suffix}"] = hex_color

    marker_style = {
        "Lam1B-weight": "o",
        "Lam3B-weight": "o",
        "Lam8B-weight": "o",
        "Smol135M-weight": "^",
        "Smol360M-weight": "^",
        "Smol1.7B-weight": "^",
    }

    labels = {
        "Lam1B-weight": "1B-weight",
        "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight",
        "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight",
        "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight",
        "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight",
        "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight",
        "Smol1.7B-tool": "1.7B-tool",
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
        zero_rows.append(
            {
                "model": model,
                "dataset_size": "0",
                "dataset_size_n": 350,  # fake 0 position for log-scale
                "tv": 0.0,
            }
        )
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
            ax.plot(x, y, color=color, label=label, marker=marker_style.get(model_label, "o"))

    ax.set_xscale("log", base=10)
    ax.set_xticks([350, 512, 1024, 2096, 8192, 50000])
    ax.set_xticklabels(["0", "500", "1k", "5k", "10k", "50k"])
    ax.set_xlabel(f"Facts Memorized (≥{recall_threshold} recall)")
    ax.set_ylabel("Total Variation")

    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="weight"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="tool baseline (worst)"),
        ],
        loc="upper left",
        # bbox_to_anchor=(0.0, -0.02),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=1.0,
        handlelength=1.35,
    )
    ax.add_artist(linestyle_legend)

    llama_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Lam1B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam3B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="3B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam8B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="8B",
        ),
    ]
    smol_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Smol135M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="135M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol360M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="360M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol1.7B-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1.7B",
        ),
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
        fontsize=7.6,
    )
    ax.add_artist(family_legend)
    ax.text(0.205, 1.13, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    ax.text(0.738, 1.13, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}_{recall_threshold}.svg"), format="svg", bbox_inches="tight")
    plt.show()


# ----- Plot Total Variation vs Training Steps (all models, specific dataset_size)
def plot_tv_vs_training_steps(df, save_path, dataset_size="10k", recall_threshold=0.9, mode="TV", save_name=None):
    fig, ax = plt.subplots(figsize=(2.15, 2.15 / 1.2))  # ICLR-style

    model_labels = ["Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"]
    cmap = cm.get_cmap("viridis", len(model_labels))
    start, end = 0.04, 0.84
    color_map = {}

    for i, model in enumerate(model_labels):
        frac = 1.0 - i / (len(model_labels) - 1)
        truncated_frac = start + frac * (end - start)
        color = cmap(truncated_frac)
        hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
        for suffix in ["-weight", "-tool"]:
            color_map[f"{model}{suffix}"] = hex_color

    marker_style = {
        "Lam1B-weight": "o",
        "Lam3B-weight": "o",
        "Lam8B-weight": "o",
        "Smol135M-weight": "^",
        "Smol360M-weight": "^",
        "Smol1.7B-weight": "^",
    }

    labels = {
        "Lam1B-weight": "1B-weight",
        "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight",
        "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight",
        "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight",
        "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight",
        "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight",
        "Smol1.7B-tool": "1.7B-tool",
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

        # ✂️ Truncate 135M at step 54
        if model_label == "Smol135M-weight":
            sub = sub[sub["checkpoint_nbr"] <= 54]

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
        marker = marker_style.get(model_label, "o")
        y_col, yerr_col = ("tv", "tv_stderr") if mode == "TV" else ("kl", "kl_stderr")

        ax.plot(
            sub["checkpoint_nbr"],
            sub[y_col],
            label=labels[model_label],
            color=color,
            linestyle="-",
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
    # ax.set_title(f"{dataset_size}")

    # Line style legend
    linestyle_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="weight"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="tool (worst)"),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.03, 0.62),
        ncol=1,
        frameon=False,
        framealpha=0.2,
        handletextpad=0.25,
        columnspacing=1.05,
        handlelength=1.1,
        # title=f"Dataset: {dataset_size} facts",
        title_fontsize=6.7,
    )
    ax.add_artist(linestyle_legend)

    # Family legend with model sizes
    llama_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Lam1B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam3B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="3B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam8B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="8B",
        ),
    ]
    smol_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Smol135M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="135M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol360M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="360M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol1.7B-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1.7B",
        ),
    ]

    # family_legend = ax.legend(
    #     handles=llama_handles + smol_handles,
    #     labels=["1B", "3B", "8B  ", "135M", "360M", "1.7B"],
    #     loc="lower center", #upper
    #     bbox_to_anchor=(0.4, 1.182),
    #     ncol=6,
    #     frameon=False,
    #     handletextpad=0.1,
    #     columnspacing=0.55,
    #     handlelength=1.2,
    #     fontsize=7.
    # )
    # ax.add_artist(family_legend)

    # ax.text(0.015, 1.16, "Llama Models", transform=ax.transAxes, ha="center", fontsize=8)
    # ax.text(0.7, 1.16, "SmolLM Models", transform=ax.transAxes, ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        save_name = "tv_vs_steps" if mode == "TV" else "kl_vs_steps"
    plt.savefig(
        os.path.join(save_path, f"{save_name}_{dataset_size}_{recall_threshold}_final.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.show()


# ----- Plot TotalVariation vs Training Steps (two plots side by side, different dataset_sizes)
def plot_tv_vs_training_steps_side_by_side(
    df, save_path, dataset_sizes=("500", "50k"), recall_threshold=0.9, mode="TV", save_name="tv_vs_steps_side_by_side"
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import cm
    import os

    fig, axes = plt.subplots(1, 2, figsize=(3.8, 3.8 / 1.6), sharey=False)

    model_labels = ["Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"]
    cmap = cm.get_cmap("viridis", len(model_labels))
    start, end = 0.04, 0.84
    color_map = {}
    for i, model in enumerate(model_labels):
        frac = 1.0 - i / (len(model_labels) - 1)
        truncated_frac = start + frac * (end - start)
        color = cmap(truncated_frac)
        hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
        for suffix in ["-weight", "-tool"]:
            color_map[f"{model}{suffix}"] = hex_color

    marker_style = {
        "Lam1B-weight": "o",
        "Lam3B-weight": "o",
        "Lam8B-weight": "o",
        "Smol135M-weight": "^",
        "Smol360M-weight": "^",
        "Smol1.7B-weight": "^",
    }

    labels = {
        "Lam1B-weight": "1B-weight",
        "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight",
        "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight",
        "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight",
        "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight",
        "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight",
        "Smol1.7B-tool": "1.7B-tool",
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
            marker = marker_style.get(model_label, "o")
            y_col, yerr_col = ("tv", "tv_stderr") if mode == "TV" else ("kl", "kl_stderr")

            ax.plot(
                sub["checkpoint_nbr"],
                sub[y_col],
                label=labels[model_label],
                color=color,
                linestyle="-",
                marker=marker,
            )
            ax.fill_between(
                sub["checkpoint_nbr"],
                sub[y_col] - sub[yerr_col],
                sub[y_col] + sub[yerr_col],
                color=color,
                alpha=0.15,
            )

        # ax.set_title(f"{dataset_size} facts")
        # ax.set_xlabel(f"Training Step (until recall ≥ {recall_threshold})")
        ax.grid(alpha=0.6)

    axes[0].set_ylabel("Total Variation" if mode == "TV" else "KL Divergence")
    fig.supxlabel(f"Training Step (until recall ≥ {recall_threshold})", y=0.11, fontsize=9)  # fonsize

    # Legend: model sizes
    llama_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Lam1B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam3B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="3B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam8B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="8B",
        ),
    ]
    smol_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Smol135M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="135M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol360M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="360M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol1.7B-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1.7B",
        ),
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
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.25, label="weight"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.25, label="tool (worst)"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.435, 0.57),
        ncol=1,
        frameon=False,  # framealpha=0.2,
        handletextpad=0.25,
        columnspacing=1.0,
        handlelength=1,
        fontsize=7.5,
    )
    axes[1].add_artist(linestyle_legend)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, f"{save_name}_{mode}_{recall_threshold}.svg"), format="svg", bbox_inches="tight"
    )
    plt.show()


# ----------    Three metrics (subplots) vs training step    ------------


def plot_recall_hellaswag_tv(df, save_path, dataset_size="10k", recall_threshold=0.9, save_name="triple_plot_vs_steps"):
    fig, axes = plt.subplots(1, 3, figsize=(7, 7 / (3 * 1.2)), sharex=False)

    model_labels = ["Smol135M", "Smol360M", "Lam1B", "Smol1.7B", "Lam3B", "Lam8B"]
    cmap = cm.get_cmap("viridis", len(model_labels))
    start, end = 0.04, 0.84
    color_map = {}
    for i, model in enumerate(model_labels):
        frac = 1.0 - i / (len(model_labels) - 1)
        truncated_frac = start + frac * (end - start)
        color = cmap(truncated_frac)
        hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
        for suffix in ["-weight", "-tool"]:
            color_map[f"{model}{suffix}"] = hex_color

    marker_style = {
        "Lam1B-weight": "o",
        "Lam3B-weight": "o",
        "Lam8B-weight": "o",
        "Smol135M-weight": "^",
        "Smol360M-weight": "^",
        "Smol1.7B-weight": "^",
    }

    labels = {
        "Lam1B-weight": "1B-weight",
        "Lam1B-tool": "1B-tool",
        "Lam3B-weight": "3B-weight",
        "Lam3B-tool": "3B-tool",
        "Lam8B-weight": "8B-weight",
        "Lam8B-tool": "8B-tool",
        "Smol135M-weight": "135M-weight",
        "Smol135M-tool": "135M-tool",
        "Smol360M-weight": "360M-weight",
        "Smol360M-tool": "360M-tool",
        "Smol1.7B-weight": "1.7B-weight",
        "Smol1.7B-tool": "1.7B-tool",
    }

    df_subset = df[df["dataset_size"] == dataset_size].copy()
    df_subset = df_subset.sort_values("checkpoint_nbr")

    for i, metric in enumerate(["recall", "hellaswag_relative", "tv"]):
        ax = axes[i]
        yerr_col = {"recall": "recall_stderr", "hellaswag_relative": "hellaswag_relative_stderr", "tv": "tv_stderr"}[
            metric
        ]
        y_label = {"recall": "Recall Accuracy", "hellaswag_relative": "HellaSwag Relative", "tv": "Total Variation"}[
            metric
        ]

        tool_df = df_subset[df_subset["model"].str.contains("tool")]
        if not tool_df.empty:
            tool_baseline = tool_df[metric].max()
            ax.axhline(tool_baseline, color="black", linestyle="--", linewidth=1.15, label="tool baseline")

        for model_label in sorted(df_subset["model"].unique()):
            if "tool" in model_label:
                continue
            sub = df_subset[df_subset["model"] == model_label].copy()
            sub = sub[sub["recall"] <= recall_threshold]
            if sub.empty:
                continue
            # if model_label == "Smol135M-weight":
            #     sub = sub[sub["checkpoint_nbr"] <= 66]

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
                marker=marker_style.get(model_label, "o"),
                linestyle="-",
            )
            ax.fill_between(
                sub["checkpoint_nbr"],
                sub[metric] - sub[yerr_col],
                sub[metric] + sub[yerr_col],
                color=color_map[model_label],
                alpha=0.15,
            )

        ax.set_xlabel("Train Step")
        ax.set_ylabel(y_label)
        # if metric == "hellaswag_relative":
        #     ax.set_ylabel("HellaSwag Perf.", labelpad=11.5)
        #     ax.text(
        #         -0.26, 0.5,  # left of axis, vertically centered
        #         "(% of base model)",
        #         fontsize=8,
        #         rotation=90,
        #         va='center',
        #         ha='right',
        #         transform=ax.transAxes
        #     )
        # else:
        #     ax.set_ylabel(y_label)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    llama_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Lam1B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam3B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="3B",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Lam8B-weight"],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="8B",
        ),
    ]
    smol_handles = [
        Line2D(
            [0],
            [0],
            color=color_map["Smol135M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="135M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol360M-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="360M",
        ),
        Line2D(
            [0],
            [0],
            color=color_map["Smol1.7B-weight"],
            marker="^",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            label="1.7B",
        ),
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
        fontsize=7.6,
    )
    fig.text(0.368, 1.065, "Llama Models", ha="center", fontsize=8)
    fig.text(0.61, 1.065, "SmolLM Models", ha="center", fontsize=8)

    # Legend for line styles
    linestyle_legend = axes[0].legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="weight"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="tool (worst)"),
        ],
        # loc="lower right",
        bbox_to_anchor=(0.5, 0.3),
        ncol=1,
        frameon=False,
        handletextpad=0.25,
        columnspacing=1.05,
        handlelength=1.2,
        fontsize=7.6,
    )
    axes[0].add_artist(linestyle_legend)

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, f"{save_name}_{dataset_size}_{recall_threshold}.svg"), format="svg", bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    # 1. Directory to save the plots
    SAVE_PATH = "/cluster/home/shouliston/DPOUncertainty/MemorySFT/Analysis"

    # 2. Indicate dir that contains 'Results_llama_final' and 'Results_smol_final'
    #       !You will have to change 'base_models_paths'
    eval_results_path = Path("/cluster/home/shouliston/DPOUncertainty/MemorySFT/Evaluation")

    # 3. Get checkpoints information
    recall_records = collect_all_recall_results(eval_results_path)
    hellaswag_records = collect_all_Hellaswag_results(eval_results_path)
    kl_tv_records = collect_all_KL_TV_results(eval_results_path)
    hellaswag_df = pd.DataFrame(hellaswag_records)
    recall_df = pd.DataFrame(recall_records)
    kl_tv_df = pd.DataFrame(kl_tv_records)

    # 4. Assemble, results in one dataframe:
    full_df = get_recall_hellaswag_tv_run_data(recall_df, kl_tv_df, hellaswag_df, eval_results_path)

    # 5. Plotting
    # -----  Hellaswag vs dataset_size
    plot_hellaswag_vs_datasetsize_absolute(
        full_df, SAVE_PATH, acc_threshold=0.95, save_name="hellaswag_vs_facts_absolute"
    )

    # -----  Hellaswag vs dataset_size
    plot_final_tv_vs_dataset_size(full_df, SAVE_PATH, recall_threshold=0.95, save_name="final_kl_vs_facts")

    # -----  Total Variation vs train_steps (mode="TV" or "KL" for Total Variaiton or KL divergence resp.)
    plot_tv_vs_training_steps(full_df, SAVE_PATH, dataset_size="500", recall_threshold=0.95, mode="TV")

    # -----  Total Variation vs train_steps (2 subplots, size by side)
    plot_tv_vs_training_steps_side_by_side(
        full_df,
        SAVE_PATH,
        dataset_sizes=("500", "50k"),
        recall_threshold=0.95,
        mode="TV",
        save_name="tv_vs_steps_side_by_side_no135",
    )

    # -----  Recall & Hellaswag & Total Variation  vs dataset_size (three subplots)
    plot_recall_hellaswag_tv(
        full_df, SAVE_PATH, dataset_size="500", recall_threshold=1.0, save_name="triple_plot_vs_steps"
    )

    # 6. Compute training steps average by each model, for each dataset:
    def aggregate_training_steps_for_latex(df):
        df_filtered = df[df["recall"] > 0.95].copy()
        df_filtered["family"] = df_filtered["model"].apply(lambda x: "in-tool" if "tool" in x else "in-weight")
        df_filtered["model_size"] = df_filtered["model"].str.extract(r"(Smol135M|Smol360M|Smol1.7B|Lam1B|Lam3B|Lam8B)")
        grouped = df_filtered.groupby(["model", "dataset_size"])
        last_ckpt = grouped["checkpoint_nbr"].max().reset_index()
        merged = pd.merge(
            last_ckpt,
            df_filtered[["model", "dataset_size", "family", "model_size"]].drop_duplicates(),
            on=["model", "dataset_size"],
            how="left",
        )

        avg_steps = merged.groupby(["family", "dataset_size"])["checkpoint_nbr"].mean().reset_index()
        avg_steps = avg_steps.pivot(index="dataset_size", columns="family", values="checkpoint_nbr").reset_index()
        return avg_steps

    # Compute the averages and save result
    result_df = aggregate_training_steps_for_latex(full_df)
    result_df.to_csv(f"{SAVE_PATH}/avg_training_steps.csv", index=False)
    print("✅ Table saved as avg_training_steps.csv and avg_training_steps.tex")
