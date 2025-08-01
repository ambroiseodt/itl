import os
import re
import json
import math
import torch
import pprint
import argparse
from tqdm import tqdm
from typing import List
from pathlib import Path
from torch.nn import functional as F
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM

def extract_step(name):
    match = re.search(r"checkpoint[-_]?(\d+)", name)
    return int(match.group(1)) if match else float('inf')

def extract_or_raise(pattern, name, string):
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract {name} from: {string}")

def group_runs_by_model(runs):
    groups = {"Lam1B": [], "Lam3B": [], "Lam8B": [], "Smol135M": [], "Smol360M": [], "Smol1.7B": []}
    for run in runs:
        if "Lam1B" in run:
            groups["Lam1B"].append(run)
        elif "Lam3B" in run:
            groups["Lam3B"].append(run)
        elif "Lam8B" in run:
            groups["Lam8B"].append(run)
        elif "Smol135M" in run:
            groups["Smol135M"].append(run)
        elif "Smol360M" in run:
            groups["Smol360M"].append(run)
        elif "Smol1.7B" in run:
            groups["Smol1.7B"].append(run)
        else:
            raise ValueError(f"Run name '{run}' does not indicate model group.")
    return groups

def get_model_name(run_name):
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
        raise ValueError("Model size (Lam1B, Lam3B, Lam8B, Smol135M, Smol360M, Smol1.7B) must be in run_name.")

def get_max_nfacts(run_names):
    max_facts = 0
    for run_name in run_names:
        nfacts = int(extract_or_raise(r"facts=(\d+)", "facts", run_name))
        max_facts = max(max_facts, nfacts)
    return max_facts

def should_skip_run(run_path, results_file_path):
    """
    Check if all checkpoints in a run directory are already evaluated.

    Returns:
        skip (bool): True if all checkpoints are already evaluated.
        existing_results (dict): Loaded results, or empty if none.
    """
    evaluated_checkpoints = set()
    existing_results = {}
    if os.path.exists(results_file_path):
        try:
            with open(results_file_path, "r") as f:
                existing_results = json.load(f)
            evaluated_checkpoints = set(existing_results.keys())
        except Exception as e:
            print(f"[!] Could not load existing results from {results_file_path}: {e}")

    all_checkpoints = sorted([
        name for name in os.listdir(run_path)
        if os.path.isdir(os.path.join(run_path, name)) and name.startswith("checkpoint")
    ], key=extract_step)

    if set(all_checkpoints).issubset(evaluated_checkpoints):
        return True, existing_results  # Skip the run

    return False, existing_results  # Still work to do


def preprocess_eval_data(eval_dataset, tokenizer, max_length=1024):
    prompts = [
        tokenizer.apply_chat_template(
            [m for m in qa if m["role"] == "user"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for qa in eval_dataset["qa"]
    ]
    tokenized = tokenizer(prompts, padding=True, truncation=True, max_length=max_length)

    # Add extra fields to original dataset
    eval_dataset = eval_dataset.add_column("user_prompt_input_ids", tokenized["input_ids"])
    eval_dataset = eval_dataset.add_column("user_prompt_attention_mask", tokenized["attention_mask"])
    eval_dataset = eval_dataset.add_column("prompt", prompts)
    eval_dataset = eval_dataset.add_column("ground_truth", [v.strip().lower() for v in eval_dataset["value"]])

    print(f"eval_dataset:\n{eval_dataset}\nFirst row: {eval_dataset[0]}")
    return eval_dataset


def batched_generate(model, tokenizer, input_ids, attention_mask, max_new_tokens=512):
    tokenizer.padding_side = "left"
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_texts = []
    for i in range(input_ids.size(0)):
        input_len = len(input_ids[i])                 
        gen_tokens = outputs[i][input_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        generated_texts.append(gen_text)

    return generated_texts


def prepare_general_prompts(tokenizer, device, prompts=None, max_length=512):
    tokenizer.padding_side = "left"
    chat_formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        ) for prompt in prompts
    ]
    encoded = tokenizer(chat_formatted, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    return {"prompts": prompts, "prompt_templated_tokenized": encoded}


def general_capability_demo_and_store(model, tokenizer, general_prompt_data, max_new_tokens=512):
    prompts = general_prompt_data["prompts"]
    encoded = general_prompt_data["prompt_templated_tokenized"]
    generated_responses = batched_generate(
        model,
        tokenizer,
        encoded["input_ids"],
        encoded["attention_mask"],
        max_new_tokens=max_new_tokens
    )

    generations = []
    for i, (prompt_text, response) in enumerate(zip(prompts, generated_responses)):
        generations.append({"prompt": prompt_text, "response": response})

    return generations


def evaluate_checkpoint_recall_weight(checkpoint_dir, dataset_with_tokenized_prompts, model, tokenizer, device, verbose_n=3, batch_size=32, max_new_tokens=64):
    total = len(dataset_with_tokenized_prompts["user_prompt_input_ids"])
    predictions = []
    all_generated_texts = []

    user_prompt_input_ids = torch.stack([
        torch.tensor(x) for x in dataset_with_tokenized_prompts["user_prompt_input_ids"]
    ]).to(device)

    user_prompt_attention_mask = torch.stack([
        torch.tensor(x) for x in dataset_with_tokenized_prompts["user_prompt_attention_mask"]
    ]).to(device)

    # Generate completions
    for i in tqdm(range(0, total, batch_size)):
        batch_input_ids = user_prompt_input_ids[i:i + batch_size]
        batch_attention_mask = user_prompt_attention_mask[i:i + batch_size]
        batch_outputs = batched_generate(model, tokenizer, batch_input_ids, batch_attention_mask, max_new_tokens)
        all_generated_texts.extend(batch_outputs)

    # Parse completions for correct answer
    correct = 0
    for j, answer in enumerate(all_generated_texts):
        gt = dataset_with_tokenized_prompts["ground_truth"][j]
        qa = dataset_with_tokenized_prompts["qa"][j]
        question = next((m["content"] for m in qa if m["role"] == "user"), "[NO USER MESSAGE FOUND]")
        is_correct = gt in answer.lower()
        correct += int(is_correct)

        if j < verbose_n:
            print(f"\n\n Eval checkpoint-{extract_step(checkpoint_dir)}")
            print(f"--- Example {j} ---")
            print(f"[Question]:\n{question}")
            print(f"[GENERATED]:\n{answer}")
            print(f"[GROUND TRUTH]: {gt}")
            print(f"[CORRECT]: {is_correct}")

        predictions.append({
            "question": question,
            "completion": answer,
            "ground_truth": gt,
            "correct": is_correct
        })

    accuracy = correct / total
    stderr = math.sqrt(accuracy * (1 - accuracy) / total)
    return accuracy, stderr, predictions


def evaluate_checkpoint_recall_tool(checkpoint_dir, dataset_with_tokenized_prompts, model, tokenizer, device, verbose_n=3, max_new_tokens=64, batch_size=8):
    predictions = []
    correct = 0

    user_prompt_input_ids = dataset_with_tokenized_prompts["user_prompt_input_ids"]
    user_prompt_attention_mask = dataset_with_tokenized_prompts["user_prompt_attention_mask"]
    user_prompt_input_ids = torch.tensor(user_prompt_input_ids, dtype=torch.long).to(device)
    user_prompt_attention_mask = torch.tensor(user_prompt_attention_mask, dtype=torch.long).to(device)

    # Step 1: Generate first-turn completions in batch
    first_turn_generations = []
    num_samples = len(dataset_with_tokenized_prompts)
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_input_ids = user_prompt_input_ids[i:i + batch_size]
        batch_attention_mask = user_prompt_attention_mask[i:i + batch_size]
        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for out, input_len in zip(batch_outputs, (batch_input_ids != tokenizer.pad_token_id).sum(dim=1)):
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            first_turn_generations.append(decoded)

    print(f"\nLets print first_turn_generations[0]:\n{first_turn_generations[0]}")

    # Step 2: Add tool-text if correct database call, and query model for final answer.
    all_conversations = []
    for j, tool_text in enumerate(first_turn_generations):
        row = dataset_with_tokenized_prompts[j]
        qa = row["qatool"]
        attribute = row["attribute"]
        value = row["value"]
        person_name = row["name"].strip().lower()
        ground_truth = str(value).strip().lower()

        user_msg = next((m["content"] for m in qa if m["role"] == "user"), None)
        if user_msg is None:
            raise ValueError(f"No user message in row {j}")

        sql_match = re.search(r"```sql\s*\nFIND\s+(\w+)\s+FOR\s+(.+?)\s*```", tool_text, re.IGNORECASE)

        if sql_match:
            requested_attr = sql_match.group(1).strip().lower()
            requested_entity = sql_match.group(2).strip().lower()
            correct_attr = attribute.strip().lower()

            assistant_tool_call = {"role": "assistant", "content": tool_text}

            if requested_attr == correct_attr and requested_entity == person_name:
                db_response = {"role": "ipython", "content": value}
            else:
                db_response = {"role": "ipython", "content": "No such record in database"}

            dialogue = [
                {"role": "user", "content": user_msg},
                assistant_tool_call,
                db_response
            ]
    
            input_ids_2 = tokenizer.apply_chat_template(dialogue, add_generation_prompt=True, return_tensors="pt").to(device)
            answer_tokens = model.generate(input_ids=input_ids_2, max_new_tokens=max_new_tokens, do_sample=False)
            final_answer = tokenizer.decode(answer_tokens[0][input_ids_2.shape[-1]:], skip_special_tokens=True).strip()
        else:
            final_answer = tool_text  # direct answer fallback

        is_correct = ground_truth in final_answer.lower()
        correct += int(is_correct)
        if j < verbose_n:
            print(f"\n\n Eval checkpoint-{extract_step(checkpoint_dir)}")
            print(f"--- Example {j} ---")
            print(f"[Question]:\n{user_msg}")
            print(f"[Tool Call]:\n{tool_text if sql_match else '[NO TOOL USED]'}")
            print(f"[DB Answer]: {db_response['content'] if sql_match else '[DIRECT]'}")
            print(f"[Final Gen]:\n{final_answer}")
            print(f"[GROUND TRUTH]: {ground_truth}")
            print(f"[CORRECT]: {is_correct}")

        predictions.append({
            "question": user_msg,
            "tool_call": tool_text if sql_match else None,
            "db_answer": db_response["content"] if sql_match else None,
            "final_answer": final_answer,
            "ground_truth": ground_truth,
            "correct": is_correct
        })

    # Evaluate all dialogues for factual recall in the assistant's final answer
    for i, convo in enumerate(all_conversations):
        continue

    accuracy = correct / num_samples
    stderr = math.sqrt(accuracy * (1 - accuracy) / num_samples)
    return accuracy, stderr, predictions


def evaluate_all_checkpoints_of_run(
        checkpoints_dir,
        results_file_path,
        general_prompt_data,
        dataset_with_tokenized_prompts_train,
        dataset_with_tokenized_prompts_test,
        verbose_n=3,
        batch_size=32,
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for name in sorted(os.listdir(checkpoints_dir), key=extract_step):
        adapter_path = os.path.join(checkpoints_dir, name)
        if os.path.isdir(adapter_path) and name.startswith("checkpoint"):
            model = AutoModelForCausalLM.from_pretrained(adapter_path, torch_dtype=torch.bfloat16).to(device)
            model.eval()

            if 'weight' in adapter_path:
                print(f"\n--Evaluating recall in weight-mode:")
                acc_train, stderr_train, preds_train = evaluate_checkpoint_recall_weight(
                    adapter_path, dataset_with_tokenized_prompts_train, model, general_prompt_data["tokenizer"], device, verbose_n=verbose_n, batch_size=batch_size
                )
                acc_test, stderr_test, preds_test = evaluate_checkpoint_recall_weight(
                    adapter_path, dataset_with_tokenized_prompts_test, model, general_prompt_data["tokenizer"], device, verbose_n=0, batch_size=batch_size
                )
            elif 'tool' in adapter_path:
                print(f"\n--Evaluating recall in tool-mode:")
                acc_train, stderr_train, preds_train = evaluate_checkpoint_recall_tool(
                    adapter_path, dataset_with_tokenized_prompts_train, model, general_prompt_data["tokenizer"], device, verbose_n=verbose_n, batch_size=batch_size
                )
                acc_test, stderr_test, preds_test = evaluate_checkpoint_recall_tool(
                    adapter_path, dataset_with_tokenized_prompts_test, model, general_prompt_data["tokenizer"], device, verbose_n=0, batch_size=batch_size
                )
            else:
                raise ValueError(f"Checkpoint path {adapter_path} doesn't contain 'weight' nor 'tool'.")

            general_gens = general_capability_demo_and_store(model, general_prompt_data["tokenizer"], general_prompt_data)

            print(f"\n ------- checkpoint-{extract_step(name)} -------")
            print(f"    Train Acc = {acc_train:.2%} ± {stderr_train:.2%}")
            print(f"    Test  Acc = {acc_test:.2%} ± {stderr_test:.2%}")
            print(f"    -- General Eval --")
            print(f"    [PROMPT]: {general_gens[0]['prompt']}")
            print(f"    [RESPONSE]: {general_gens[0]['response']}")

            results[name] = {
                "num_eval_train": len(dataset_with_tokenized_prompts_train["user_prompt_input_ids"]),
                "accuracy_train": acc_train,
                "stderr_train": stderr_train,
                "num_eval_test": len(dataset_with_tokenized_prompts_test["user_prompt_input_ids"]),
                "accuracy_test": acc_test,
                "stderr_test": stderr_test,
                "general_capabilities": general_gens,
                "predictions_train": preds_train[:5],
                "predictions_test": preds_test[:5],
            }

    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing models/checkpoints to evaluate.")
    parser.add_argument("--base_results_dir", type=str, default=None, help="Directory to save evaluation results.")
    args = parser.parse_args()

    # Set models dir and saving dir 
    EXPERIMENTS_DIR = args.models_dir
    if args.base_results_dir: 
        EVALS_DIR = f"{args.base_results_dir}/Recall"
    else: 
        EVALS_DIR = Path(__file__).parents[1] / "Results" / "Recall"  
    os.makedirs(EVALS_DIR, exist_ok=True)

    DATA_DIR = Path(__file__).parents[2] / "Data" / "HF_dataset_200_000"

    # Samples to test on
    N_EVAL_TRAIN = 300
    N_EVAL_TEST = 100

    general_prompts = [
        "Explain the concept of gravity to a 5-year-old.",
        "What are the pros and cons of remote work?",
        "Tell me a short story about a robot who learns to feel emotions.",
    ]

    # Group runs depending on base model (in case we load LoRa adapters)
    all_runs = sorted([r for r in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, r))])
    grouped_runs = group_runs_by_model(all_runs)
    full_dataset = Dataset.load_from_disk(DATA_DIR)
    torch.manual_seed(42)

    for group, run_names in grouped_runs.items():
        if not run_names:
            continue
        print(f"\n============      Evaluating group {group}     ===============")

        model_name = get_model_name(run_names[0])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "llama" in model_name.lower():      
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.padding_side = "left"

        # Tokenize the factual recall user prompts
        print(f"\nTokenizing HF_dataset")
        max_nfacts = get_max_nfacts(run_names) + N_EVAL_TEST
        subset = full_dataset.select(list(range(0, min(len(full_dataset), max_nfacts))))
        dataset_with_tokenized_prompts = preprocess_eval_data(subset, tokenizer)

        # Tokenize the general prompts
        print(f"\nTokenizing general prompts")
        general_prompt_data = {
                    "tokenizer": tokenizer,
                    "prompts": general_prompts,
                    "prompt_templated_tokenized": tokenizer(
                        [tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True
                        ) for p in general_prompts],
                        return_tensors="pt", padding=True, truncation=True, max_length=512
                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                }

        for run_name in run_names:
            run_path = os.path.join(EXPERIMENTS_DIR, run_name)
            eval_output_dir = os.path.join(EVALS_DIR, run_name)
            results_file_path = os.path.join(eval_output_dir, "checkpoints_recall_results.json")
            should_skip, existing_results = should_skip_run(run_path, results_file_path)
            if should_skip:
                print(f"[✓] Skipping {run_name} — all checkpoints already evaluated.")
                continue

            print(f"\n================= Evaluating run: {run_name} =================")
            # Slice the dataset indices appropriately for this run
            n_facts = int(extract_or_raise(r"facts=(\d+)", "facts", run_name))
            train_indices = torch.randperm(n_facts)[:N_EVAL_TRAIN].tolist()
            test_indices = list(range(n_facts, min(n_facts + N_EVAL_TEST, len(subset))))

            dataset_with_tokenized_prompts_train = dataset_with_tokenized_prompts.select(train_indices) 
            dataset_with_tokenized_prompts_test = dataset_with_tokenized_prompts.select(test_indices)
                        
            for batch_size in [128, 64, 32, 24, 16, 8, 4]:
                try:
                    print(f"[→] Evaluating with batch size {batch_size}")
                    evaluate_all_checkpoints_of_run(
                        checkpoints_dir=run_path,
                        results_file_path=results_file_path,
                        general_prompt_data=general_prompt_data,
                        dataset_with_tokenized_prompts_train=dataset_with_tokenized_prompts_train,
                        dataset_with_tokenized_prompts_test=dataset_with_tokenized_prompts_test,
                        verbose_n=1,
                        batch_size=batch_size,
                    )
                    break  # exit loop if successful

                except Exception as e:
                    print(f"[!] Failed with batch size {batch_size}: {e}")
                    print(f"[✗] Skipping run {run_name} due to error: {e}")
                    
    print(f"\n\n\nEvaluation Finished.")