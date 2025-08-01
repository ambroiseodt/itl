import argparse
import json
import os
import re
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_step(name):
    match = re.search(r"checkpoint[-_]?(\d+)", name)
    return int(match.group(1)) if match else float("inf")


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
        return "/cluster/home/shouliston/DPOUncertainty/SmolLM-360M_Instruct"  # Local folder
    elif "Smol1.7B" in run_name:
        return "HuggingFaceTB/SmolLM-1.7B-Instruct"
    else:
        raise ValueError("Model size (Lam1B, Lam3B, Lam8B, Smol135M, Smol360M, Smol1.7B) must be in run_name.")


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

    all_checkpoints = sorted(
        [
            name
            for name in os.listdir(run_path)
            if os.path.isdir(os.path.join(run_path, name)) and name.startswith("checkpoint")
        ],
        key=extract_step,
    )

    if set(all_checkpoints).issubset(evaluated_checkpoints):
        return True, existing_results  # Skip the run

    return False, existing_results  # Still work to do


# -------------------- KL Divergence Functions --------------------


def find_subsequence(sequence, subseq):
    """
    Returns the index immediately after a subsequence `subseq` inside `sequence`.
    If not found, returns -1.
    """
    for i in range(len(sequence) - len(subseq) + 1):
        if torch.equal(sequence[i : i + len(subseq)], subseq):
            return i + len(subseq)
    return -1


def process_generated_batch(output, tokenizer, device, model_family):
    """
    Processes generate() outputs into:
    - full input_ids (including left/right padding)
    - labels (masked to only apply on generated completion tokens)

    Returns a dict with padded tensors ready for KL divergence computation.
    """
    # print(f"\n\n==============\n    Entering process_generated_batch\n =================\n")
    input_ids_list, labels_list = [], []
    full_input_ids = output["sequences"]  # [B, T_total]

    assistant_marker = get_assistant_marker(tokenizer, model_family)
    assistant_marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
    assistant_marker_tensor = torch.tensor(assistant_marker_ids).to(device)

    for i in range(full_input_ids.size(0)):
        seq = full_input_ids[i]  # [T_total]
        start_idx = find_subsequence(seq, assistant_marker_tensor)
        # Determine where padding starts (i.e. end of actual content)
        non_pad = (seq != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        end_idx = non_pad[-1].item()  # exclusive

        # Full sequence
        input_ids = seq

        # Only label completion portion
        labels = torch.full_like(input_ids, -100)
        labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list).to(device),
        "labels": torch.stack(labels_list).to(device),
    }


def generate_with_logits(model, tokenizer, prompts: list[str], model_family, max_new_tokens=400, batch_size=4):
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    all_sequences = []
    max_seq_len = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        chat_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch_prompts
        ]

        tokenized_prompts = tokenizer(
            chat_prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **tokenized_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
            )

        all_sequences.append(output.sequences)
        max_seq_len = max(max_seq_len, output.sequences.shape[1])

    # Pad all sequences to the global max_seq_len
    padded_sequences = [
        torch.nn.functional.pad(seq, (0, max_seq_len - seq.shape[1]), value=pad_token_id)
        if seq.shape[1] < max_seq_len
        else seq
        for seq in all_sequences
    ]

    combined_output = {
        "sequences": torch.cat(padded_sequences, dim=0),
    }
    return process_generated_batch(combined_output, tokenizer, device, model_family)


def selective_log_softmax(logits, labels, tokenizer=None):
    """
    For each token in `labels`, computes log softmax from logits.
    Ignores tokens where labels == -100.

    Args:
        logits: [B, T, V] tensor of model logits
        labels: [B, T] labels token ids
        tokenizer: for optional debugging

    Returns:
        logprobs: [B, T] tensor of logprobs at labels positions
    """
    B, T, V = logits.shape
    assert labels.shape == (B, T)

    mask = labels != -100
    labels_clamped = labels.clone()
    labels_clamped[~mask] = 0  # dummy safe index

    # Debug any token index errors
    invalid = (labels_clamped >= V) & mask
    if invalid.any():
        b, t = torch.nonzero(invalid, as_tuple=True)
        for i in range(len(b)):
            bad_id = labels[b[i], t[i]].item()
            decoded = tokenizer.decode([bad_id]) if tokenizer else "<unknown>"
            print(f"[ERROR] Token ID {bad_id} at batch={b[i]}, pos={t[i]} ≥ vocab {V}: '{decoded}'")
        raise ValueError(f"Found token ID ≥ vocab size ({V})")

    logprobs = []
    for row_logits, row_index in zip(logits, labels_clamped):
        row_log_softmax = F.log_softmax(row_logits, dim=-1)
        row_logprobs = row_log_softmax.gather(dim=-1, index=row_index.unsqueeze(-1)).squeeze(-1)
        logprobs.append(row_logprobs)

    logprobs = torch.stack(logprobs, dim=0)
    logprobs[~mask] = float("-inf")

    return logprobs


def compute_kl_from_logprobs(
    checkpoint_model,
    ref_model,
    ref_data,
    tokenizer,
    estimator: str = "standard",
    batch_size=1,
    compute_tv=True,
):
    """
    Computes KL divergence and Total Variation (TV) distance between a checkpoint model and a reference model.

    Args:
        checkpoint_model: model whose output is compared to the reference model
        ref_model: reference model
        ref_data: dictionary with keys: 'input_ids', 'labels'
        tokenizer: for optional decoding
        estimator: one of "standard", "log_square", "bregman"
        batch_size: number of samples per forward pass

    Returns:
        (mean KL, KL stderr, mean TV, TV stderr)
    """
    checkpoint_model.eval()
    ref_model.eval()
    device = next(checkpoint_model.parameters()).device

    input_ids = ref_data["input_ids"]
    labels = ref_data["labels"]
    kl_vals = []
    tv_vals = []

    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i : i + batch_size].to(device)
        batch_labels = labels[i : i + batch_size].to(device)
        mask = batch_labels != -100  # [B, T]

        with torch.no_grad():
            # Full logits
            checkpoint_logits = checkpoint_model(batch_input_ids).logits  # [B, T, V]
            ref_logits = ref_model(batch_input_ids).logits  # [B, T, V]

            # Log-probabilities
            chkpt_logprobs_full = F.log_softmax(checkpoint_logits, dim=-1)  # [B, T, V]
            ref_logprobs_full = F.log_softmax(ref_logits, dim=-1)  # [B, T, V]

            # Probabilities
            ref_probs = ref_logprobs_full.exp()  # [B, T, V]

            # ========== KL Estimator Options ==========
            if estimator == "standard":
                kl_per_token = (ref_probs * (ref_logprobs_full - chkpt_logprobs_full)).sum(dim=-1)  # [B, T]
            elif estimator == "log_square":
                diff = ref_logprobs_full - chkpt_logprobs_full
                kl_per_token = 0.5 * (ref_probs * diff.pow(2)).sum(dim=-1)  # [B, T]
            elif estimator == "bregman":
                d = chkpt_logprobs_full - ref_logprobs_full
                kl_per_token = (ref_probs * (d.exp() - 1 - d)).sum(dim=-1)  # [B, T]
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

            kl_vals.extend(kl_per_token[mask].tolist())  # [N]

            # ========== Total Variation ==========
            if compute_tv:
                chkpt_probs = chkpt_logprobs_full.exp()
                tv_per_token = 0.5 * (ref_probs - chkpt_probs).abs().sum(dim=-1)  # [B, T]
                tv_vals.extend(tv_per_token[mask].tolist())

    def summarize(vals):
        t = torch.tensor(vals)
        mean = t.mean().item()
        stderr = t.std(unbiased=True).item() / (len(t) ** 0.5) if len(t) > 1 else None
        return mean, stderr

    kl_mean, kl_stderr = summarize(kl_vals)
    print(f"\n[✓] Final KL: {kl_mean:.6f}" + (f" ± {kl_stderr:.6f}" if kl_stderr else ""))

    if compute_tv:
        tv_mean, tv_stderr = summarize(tv_vals)
        print(f"[✓] Final TV: {tv_mean:.6f}" + (f" ± {tv_stderr:.6f}" if tv_stderr else ""))
        return kl_mean, kl_stderr, tv_mean, tv_stderr

    return kl_mean, kl_stderr, None, None


def evaluate_all_checkpoints_of_run(
    checkpoints_dir,
    evals_dir,
    kl_prompts,
    base_model_name,
    batch_size=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = os.path.basename(checkpoints_dir)

    # Get tokenizer from checkpoint
    ref_checkpoint_name = sorted(os.listdir(checkpoints_dir), key=extract_step)[0]
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoints_dir, ref_checkpoint_name))
    tokenizer.padding_side = "left"
    print_special_tokens_info_new(tokenizer)

    # Determine the right special tokens
    model_family = get_model_family(run_name)
    if model_family == "Llama":
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif model_family == "Smol":
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    else:
        raise ValueError(f"model_family {model_family} should either be 'Llama' or 'Smol';")

    print_special_tokens_info_new(tokenizer)

    # Instantiate ref model for KL divergence
    ref_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16).to(device)
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.eos_token_id = tokenizer.eos_token_id
    ref_model.eval()
    if model_family == "Smol":
        ref_model.resize_token_embeddings(
            len(tokenizer)
        )  # because we're using the checkpoint tokenizer which was trained with new pad token

    # Generate the reference model completions and logits given the prompts
    ref_data = generate_with_logits(ref_model, tokenizer, kl_prompts, model_family)

    results = {}
    for name in sorted(os.listdir(checkpoints_dir), key=extract_step):  # name = checkpoint-x
        adapter_path = os.path.join(checkpoints_dir, name)
        if os.path.isdir(adapter_path) and name.startswith("checkpoint"):
            print(f"\n\n======== checkpoint-{extract_step(name)} ============")
            model = AutoModelForCausalLM.from_pretrained(adapter_path, torch_dtype=torch.bfloat16).to(device)
            model.eval()

            # Compute KL and TV
            kl_mean = kl_stderr = tv_mean = tv_stderr = None
            kl_mean, kl_stderr, tv_mean, tv_stderr = compute_kl_from_logprobs(
                model, ref_model, ref_data, tokenizer, estimator="bregman", batch_size=batch_size, compute_tv=True
            )
            results[name] = {
                "kl_mean": kl_mean,
                "kl_stderr": kl_stderr,
                "tv_mean": tv_mean,
                "tv_stderr": tv_stderr,
                "num_eval_kl": len(kl_prompts),
            }

    os.makedirs(evals_dir, exist_ok=True)
    with open(os.path.join(evals_dir, "checkpoint_kl_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------
# Adapting special tokens to model familey (llama, smol, etc.. )
def get_model_family(run_name):
    if "lam" in run_name.lower():
        model_family = "Llama"
    elif "smol" in run_name.lower():
        model_family = "Smol"
    else:
        raise ValueError(f"run_name {run_name} provided should contain 'Lam' or 'Smol' to determine model_family.")
    return model_family


def get_assistant_marker(tokenizer, model_family):
    if model_family == "Smol":
        marker = "<|im_start|>assistant"
    elif model_family == "Llama":
        marker = "<|start_header_id|>assistant<|end_header_id|>"
    else:
        raise ValueError(f"Wrong model_family {model_family} provided. Should be 'Llama' or 'Smol'.")
    return marker


def print_special_tokens_info_new(tokenizer):
    special_tokens = {
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token,
        "sep_token": getattr(tokenizer, "sep_token", None),
        "cls_token": getattr(tokenizer, "cls_token", None),
        "mask_token": getattr(tokenizer, "mask_token", None),
        "eot_token": getattr(tokenizer, "eot_token", None),  # may not exist
    }
    print("\n--- Special Tokens ---")
    for name, token in special_tokens.items():
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{name}: '{token}' | ID: {token_id}")
    print("----------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir", type=str, required=True, help="Directory containing models/checkpoints to evaluate."
    )
    parser.add_argument(
        "--base_results_dir", type=str, default=None, help="Directory to save (all) evaluation results."
    )
    args = parser.parse_args()

    # Set models dir and saving dir
    EXPERIMENTS_DIR = args.models_dir
    if args.base_results_dir:
        EVALS_DIR = f"{args.base_results_dir}/KL"
    else:
        EVALS_DIR = Path(__file__).parents[1] / "Results" / "KL"
    os.makedirs(EVALS_DIR, exist_ok=True)

    # Factual/Encyclopedic knowledge:
    kl_prompts = [
        "What is the capital of Argentina?",
        "Explain how a black hole forms in simple terms.",
        "What is the difference between mitosis and meiosis?",
        "Name three major causes of World War I.",
        "Who was Ada Lovelace and why is she important?",
        "What is the chemical formula for water?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the tallest mountain in the world?",
        "What is the main function of red blood cells?",
        "When did the Berlin Wall fall?",
        "What causes tides on Earth?",
        "Who painted the ceiling of the Sistine Chapel?",
        "What does DNA stand for?",
        "In what year did the French Revolution begin?",
        "What is the currency used in Japan?",
        "What is the largest desert in the world?",
        "Who discovered penicillin?",
        "Name the three branches of the U.S. government.",
        "What is the speed of light in a vacuum?",
        "Which planet has the most moons?",
        "What was the purpose of the Great Wall of China?",
        "What is photosynthesis?",
        "Name the primary language spoken in Brazil.",
        "Who developed the theory of general relativity?",
        "What organ in the human body is responsible for filtering blood?",
        "What year did the Titanic sink?",
        "What is the main ingredient in glass?",
        "Who was the first person to walk on the moon?",
        "What is the boiling point of water in Celsius?",
        "Name a country that is both in Europe and Asia.",
    ]

    #  Instruction Following / Creative Generation
    kl_prompts += [
        "Write a short story about a time-traveling historian.",
        "Explain the rules of chess to a beginner.",
        "Generate a haiku about artificial intelligence.",
        "Translate the following English sentence into French: 'The cat is on the table.'",
        "List 5 tips for staying productive while working from home.",
        "Invent a new holiday and describe how it is celebrated.",
        "Write a dialogue between a robot and a confused human.",
        "Create a recipe for a fictional space snack.",
        "Describe your favorite childhood memory in vivid detail.",
        "Explain how to build a paper airplane that flies well.",
        "Give instructions for making a peanut butter and jelly sandwich.",
        "Compose a short motivational speech for students.",
        "Describe the interior of an alien spaceship.",
        "Design a new board game and explain how it works.",
        "Write a poem about forgetting.",
        "Write a bedtime story for a child afraid of monsters.",
        "Create a to-do list for someone planning a surprise party.",
        "Write a product description for a magical umbrella.",
        "Compose a limerick about a cat on a skateboard.",
        "Describe a futuristic classroom in 100 words.",
        "Give a pep talk to someone about to take an important exam.",
        "Design a magical creature and explain its powers.",
        "Write a scene from a play where two pirates argue over treasure.",
        "Invent a new kind of sandwich and describe its ingredients.",
        "Write a letter from a dragon applying for a job.",
        "Explain how to teach a dog to play fetch.",
        "Make up a myth about how the moon got its craters.",
        "Write a Yelp review for a café run by robots.",
        "Describe a dream vacation to a floating island.",
        "Give advice to someone who just moved to a new city.",
    ]

    #  Cognitive Tasks / Math / Logic
    kl_prompts += [
        "What is the next number in the sequence: 2, 4, 8, 16, ...?",
        "If a car travels at 60 km/h for 2 hours, how far does it go?",
        "Describe the Monty Hall problem and the best strategy.",
        "What is the square root of 121?",
        "Explain why the sum of two even numbers is always even.",
        "What is the derivative of x^2?",
        "Convert 1101 from binary to decimal.",
        "You flip a fair coin 3 times. What is the probability of 2 heads?",
        "What does it mean for a function to be continuous?",
        "Solve for x in the equation: 3x + 2 = 11",
        "What is the prime factorization of 84?",
        "If a rectangle has sides 5 cm and 12 cm, what is its area?",
        "Explain the difference between permutations and combinations.",
        "What is the median of the set: [3, 7, 9, 15, 20]?",
        "If 3 pens cost $2.40, how much does 1 pen cost?",
        "How many degrees are in the interior angles of a triangle?",
        "What is the cube root of 64?",
        "If a die is rolled, what’s the probability of rolling a 5?",
        "What is the least common multiple of 6 and 8?",
        "Explain why dividing by zero is undefined.",
    ]

    # Language, Style, Translation
    kl_prompts += [
        "Translate 'break the ice' into French and explain it.",
        "Rewrite this formally: 'Can you give me a hand?'",
        "Paraphrase this legal phrase: 'hold harmless and indemnify.'",
        "Correct all errors: 'She go to the market and buy apples.'",
        "Summarize the tone of a Shakespeare soliloquy in one sentence.",
        "Translate 'The river sleeps under stars' into Arabic.",
        "Rewrite this sarcastically: 'Great job crashing the server.'",
        "Write a simile for memory loss (no tech metaphors).",
        "Turn this into a tweet: 'Please submit the report by noon.'",
        "Translate 'Knowledge is power' into Latin.",
        "Make this gender-neutral: 'Each player took his turn.'",
        "Change this passive voice: 'The prize was won by Sam.'",
        "Write a limerick using 'syntax', 'glitch', and 'latte'.",
        "Translate 'Ask me anything' into Japanese.",
        "Explain the idiom 'spill the tea'.",
        "Rewrite this dramatically: 'He opened the door.'",
        "List 3 synonyms for 'ephemeral' and use one in a sentence.",
        "Summarize 'Romeo and Juliet' in a tweet.",
        "Write a pun about libraries.",
        "Compare Hemingway and Woolf’s styles in one line.",
    ]

    for run_name in sorted(os.listdir(EXPERIMENTS_DIR)):
        run_path = os.path.join(EXPERIMENTS_DIR, run_name)
        if not os.path.isdir(run_path):
            continue
        print(f"\n\n================= Evaluating run: {run_name} =================")
        eval_output_dir = os.path.join(EVALS_DIR, run_name)
        os.makedirs(eval_output_dir, exist_ok=True)

        # Check if run has already been evaluated:
        results_file_path = os.path.join(eval_output_dir, "checkpoints_recall_results.json")
        should_skip, existing_results = should_skip_run(run_path, results_file_path)
        if should_skip:
            print(f"[✓] Skipping {run_name} — all checkpoints already evaluated.")
            continue

        # Perform Evaluation
        base_model_name = get_model_name(run_name)

        for batch_size in [32, 24, 16, 8, 4]:
            try:
                print(f"[→] Evaluating with batch size {batch_size}")
                results = evaluate_all_checkpoints_of_run(
                    checkpoints_dir=run_path,
                    evals_dir=eval_output_dir,
                    kl_prompts=kl_prompts,
                    base_model_name=base_model_name,
                    batch_size=batch_size,
                )
                break  # exit loop if successful

            except Exception as e:
                print(f"[!] Failed with batch size {batch_size}: {e}")
                print(f"[✗] Skipping run {run_name} due to error: {e}")
                traceback.print_exc()

    print(f"\n\n\nEvaluation Finished.")
