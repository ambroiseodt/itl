# This source code is licensed under the terms specified in the `LICENSE` file.
"""
Evaluation with Hellaswag performance.

@ 2025, Meta
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

from tqdm import tqdm

HF_DATASET_PATH = Path(__file__).parents[1] / "HF_datasets"
SAVE_PATH = Path(__file__).parents[1] / "runs"
EVAL_PATH = Path(__file__).parents[1] / "eval_runs"


def get_nice_base_model_name(model_name):
    return model_name.split("/")[-1]


def extract_step(name):
    match = re.search(r"checkpoint[-_]?(\d+)", name)
    return int(match.group(1)) if match else float("inf")


def evaluation_already_exists(output_dir: str) -> bool:
    """
    Checks if any .json file exists under output_dir.
    """
    for root, _, files in os.walk(output_dir):
        if any(f.endswith(".json") for f in files):
            return True
    return False


def get_base_model_for_peft(run_name: str) -> str:
    """
    Maps a run name to its corresponding Hugging Face base model name.
    """
    if "Lam1B" in run_name:
        return "meta-llama/Llama-3.2-1B-Instruct"
    elif "Lam3B" in run_name:
        return "meta-llama/Llama-3.2-3B-Instruct"
    elif "Lam8B" in run_name:
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif "Smol135M" in run_name:
        return "HuggingFaceTB/SmolLM-135M-Instruct"
    elif "Smol360M" in run_name:
        return "HuggingFaceTB/SmolLM-360M-Instruct"
    elif "Smol1.7B" in run_name:
        return "HuggingFaceTB/SmolLM-1.7B-Instruct"
    else:
        raise ValueError(
            f"run_name {run_name} must contain (Lam1B, Lam3B, Lam8B, Smol135M, Smol360M, Smol1.7B) must be in run_name."
        )


def is_lora_checkpoint(checkpoint_path):
    files = os.listdir(checkpoint_path)
    has_adapter_model = any("adapter_model" in f for f in files)
    has_adapter_config = "adapter_config.json" in files
    return has_adapter_model and has_adapter_config


def evaluate_peft_model_on_hellaswag(
    adaptor_path, output_path, base_model_name, batch_size="32", task_name="hellaswag"
):
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={base_model_name},peft={adaptor_path},tokenizer={adaptor_path}",
        "--tasks",
        task_name,
        "--batch_size",
        str(batch_size),
        "--output_path",
        output_path,
    ]
    print(f"[→] Running LLM Harness on {adaptor_path}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[✗] Failed on {adaptor_path}: {e}")


def evaluate_model_on_hellaswag(model_name, output_path, batch_size="32", task_name="hellaswag"):
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_name}",
        "--tasks",
        task_name,
        "--batch_size",
        str(batch_size),
        "--output_path",
        output_path,
    ]
    print(f"[→] Running LLM Harness on {model_name}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[✗] Failed on {model_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["base", "checkpoints", "both"], required=True, help="Which model type to evaluate"
    )
    parser.add_argument(
        "--base_model_family",
        choices=["llama", "smollm"],
        required=True,
        help="Specify whether model is from Llama or SmolLM family.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory containing models/checkpoints to evaluate."
    )
    parser.add_argument("--eval_dir", type=str, default=None, help="Directory to save evaluation results.")
    parser.add_argument("--target", help="Identificator to filter runs (if target in run_name)")
    args = parser.parse_args()

    # Set paths
    exp_dir = SAVE_PATH / f"{args.model_dir}"
    eval_dir = EVAL_PATH / f"{args.eval_dir}" / "hellaswag"
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    # -------    Evaluation of base models  -------
    if args.mode in ["base", "both"]:
        match args.base_model_family.lower():
            case "llama":
                base_model_names = [
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "meta-llama/Llama-3.1-8B-Instruct",
                ]
            case "smollm":
                base_model_names = [
                    "HuggingFaceTB/SmolLM-135M-Instruct",
                    "HuggingFaceTB/SmolLM-360M_Instruct",
                    "HuggingFaceTB/SmolLM-1.7B-Instruct",
                ]
            case _:
                raise ValueError(f"Invalid model family {args.base_model_family}. Options are 'llama' or 'smollm")
        for base_model_name in base_model_names:
            print(f"\n================= Evaluating run: {base_model_name} =================")
            output_path = os.path.join(eval_dir, "base_models", get_nice_base_model_name(base_model_name))
            os.makedirs(output_path, exist_ok=True)
            for batch_size in [64, 32, 24, 16, 8, 4]:
                try:
                    print(f"[→] Evaluating {base_model_name} with batch size {batch_size}")
                    evaluate_model_on_hellaswag(
                        model_name=base_model_name, output_path=output_path, batch_size=batch_size
                    )
                    break  # exit loop if successful
                except Exception as e:
                    print(f"[!] Failed with batch size {batch_size}: {e}")

    # -------    Evaluation of checkpoints  --------
    elif args.mode in ["checkpoints", "both"]:
        run_names = sorted(
            [
                d
                for d in os.listdir(exp_dir)
                if os.path.isdir(os.path.join(exp_dir, d)) and (args.target in d if args.target else True)
            ]
        )

        for run_name in tqdm(run_names, desc="Evaluating runs"):
            print(f"\n================= Evaluating run: {run_name} =================")
            run_path = os.path.join(exp_dir, run_name)
            base_model_name = get_base_model_for_peft(run_name)

            all_checkpoints = sorted(
                [
                    name
                    for name in os.listdir(run_path)
                    if os.path.isdir(os.path.join(run_path, name)) and name.startswith("checkpoint")
                ],
                key=extract_step,
            )

            for checkpoint_name in all_checkpoints:
                adaptor_path = os.path.join(run_path, checkpoint_name)
                output_path = os.path.join(eval_dir, "runs", run_name, checkpoint_name)
                os.makedirs(output_path, exist_ok=True)

                if evaluation_already_exists(output_path):
                    print(f"[✓] Skipping {run_name}-{checkpoint_name} — already evaluated.")
                    continue

                for batch_size in [256, 128, 64, 32, 24, 16, 8, 4]:
                    try:
                        print(f"\n[→] Evaluating {checkpoint_name} with batch size {batch_size}")

                        if is_lora_checkpoint(adaptor_path):
                            print(f"[✓] Detected LoRA adapter in {adaptor_path}")
                            evaluate_peft_model_on_hellaswag(
                                adaptor_path=adaptor_path,
                                output_path=output_path,
                                base_model_name=base_model_name,
                                batch_size=batch_size,
                            )
                        else:
                            print(f"[✓] Detected full model in {adaptor_path}")
                            evaluate_model_on_hellaswag(
                                model_name=adaptor_path, output_path=output_path, batch_size=batch_size
                            )
                        break
                    except Exception as e:
                        print(f"[!] Failed with batch size {batch_size}: {e}")

    else:
        raise ValueError("Wrong mode provided. Should be either 'base', 'checkpoints', or 'both'.")

    print("\n\n Hellaswag evals finished")
