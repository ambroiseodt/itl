# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Module to aggregate the experimental results into csv files.

@ 2025, Ambroise Odonnat
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESULT_PATH = Path(__file__).parents[3] / "results"

# ------------------------------------------------------------------------------
# Load and create dataframe of results from json files
# ------------------------------------------------------------------------------


def collect_all_recall_results(base_path: Path) -> list:
    """Collect all the eval results for hellaswag, recall and kl."""
    all_records = []
    for subdir in base_path.glob("Results_*_final"):  # change this to Results
        recall_dir = subdir / "Recall"
        if recall_dir.exists() and recall_dir.is_dir():
            print(f"[Info] Found recall dir: {recall_dir}")
            records = collect_recall_results(recall_dir)
            all_records.extend(records)
        else:
            print(f"[Skip] No recall directory in {subdir}")
    return all_records


def collect_recall_results(base_path: Path) -> list:
    """Collect eval results for hellaswag, recall and kl."""
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
    for _, recs in records_by_run.items():
        sorted_recs = sorted(recs, key=lambda x: x["checkpoint"])
        for epoch_idx, rec in enumerate(sorted_recs, start=1):
            rec["epoch"] = epoch_idx
            final_records.append(rec)
    return final_records


def collect_all_Hellaswag_results(base_path: Path) -> list:
    """Collect all the eval results for hellaswag."""
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


def collect_hellaswag_results(base_path: Path) -> list:
    """Collect the eval results for hellaswag."""
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
    for _, recs in records_by_run.items():
        sorted_recs = sorted(recs, key=lambda x: x["checkpoint"])
        for epoch_idx, rec in enumerate(sorted_recs, start=1):
            rec["epoch"] = epoch_idx
            final_records.append(rec)
    return final_records


def collect_all_KL_TV_results(base_path: Path) -> list:
    """Collect all the eval results for KL and TV."""
    all_records = []
    for subdir in base_path.glob("Results_*_final"):
        kl_dir = subdir / "KL"
        if kl_dir.exists() and kl_dir.is_dir():
            print(f"[Info] Found kl dir: {kl_dir}")
            records = collect_KL_TV_results(kl_dir)
            all_records.extend(records)
        else:
            print(f"[Skip] No recall directory in {subdir}")
    return all_records


def collect_KL_TV_results(base_path: Path) -> list:
    """Collect the eval results for KL and TV."""
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
    for _, recs in records_by_run.items():
        sorted_recs = sorted(recs, key=lambda x: x["checkpoint"])
        for epoch_idx, rec in enumerate(sorted_recs, start=1):
            rec["epoch"] = epoch_idx
            final_records.append(rec)

    return final_records


# ------------------------------------------------------------------------------
# Fetch specific results from the paper experiments
# ------------------------------------------------------------------------------


def get_recall_hellaswag_tv_run_data(
    recall_df: pd.DataFrame,
    tv_df: pd.DataFrame,
    hellaswag_df: pd.DataFrame,
    eval_results_path: Path,
    acc_threshold: float = 1,
) -> pd.DataFrame:
    """Fetch base model hellaswag data."""
    llama1B_path = f"{eval_results_path}/Results_llama_final/Hellaswag/base_models/Llama-3.2-1B-Instruct"
    llama3B_path = f"{eval_results_path}/Results_llama_final/Hellaswag/base_models/Llama-3.2-3B-Instruct"
    llama8B_path = f"{eval_results_path}/Results_llama_final/Hellaswag/base_model/Llama-3.1-8B-Instruct"
    smollm135M_path = f"{eval_results_path}/Results_smol_final/Hellaswag/base_models/SmolLM-135M-Instruct"
    smollm360M_path = f"{eval_results_path}/Results_smol_final/Hellaswag/base_models/SmolLM-360M_Instruct"
    smollm1700M_path = f"{eval_results_path}/Results_smol_final/Hellaswag/base_models/SmolLM-1.7B-Instruct"
    base_models_paths = {
        "Lam1B": f"{llama1B_path}/results_2025-05-06T23-30-06.017723.json",
        "Lam3B": f"{llama3B_path}/results_2025-05-07T00-07-24.007108.json",
        "Lam8B": f"{llama8B_path}/meta-llama__Llama-3.1-8B-Instruct/results_2025-05-19T10-57-37.932935.json",
        "Smol135M": f"{smollm135M_path}/HuggingFaceTB__SmolLM-135M-Instruct/results_2025-06-28T12-52-52.265790.json",
        "Smol360M": f"{smollm360M_path}/HuggingFaceTB__SmolLM-360M-Instruct/results_2025-06-28T05-07-18.025099.json",
        "Smol1.7B": f"{smollm1700M_path}/HuggingFaceTB__SmolLM-1.7B-Instruct/results_2025-06-28T05-09-42.516537.json",
    }

    base_scores = {}
    for model_id, path in base_models_paths.items():
        try:
            with open(path) as f:
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


if __name__ == "__main__":
    # 1. Directory to save the results
    if not RESULT_PATH.exists():
        RESULT_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Indicate dir that contains 'Results_llama_final' and 'Results_smol_final' (json results)
    eval_results_path = Path(__file__).parents[2] / "Evaluation"

    # 3. Get checkpoints information
    recall_records = collect_all_recall_results(eval_results_path)
    hellaswag_records = collect_all_Hellaswag_results(eval_results_path)
    kl_tv_records = collect_all_KL_TV_results(eval_results_path)
    hellaswag_df = pd.DataFrame(hellaswag_records)
    recall_df = pd.DataFrame(recall_records)
    kl_tv_df = pd.DataFrame(kl_tv_records)

    # 4. Assemble, results in one dataframe:
    full_df = get_recall_hellaswag_tv_run_data(recall_df, kl_tv_df, hellaswag_df, eval_results_path)

    # Save results:
    full_df.to_csv(f"{RESULT_PATH}/large_scale_results.csv")
