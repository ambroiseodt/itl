import os
import re
import sys
import json
import wandb
import torch
import argparse
from tqdm import tqdm
from math import ceil
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)

from accelerate import Accelerator
from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Training.ToolDataCollator import DataCollatorForToolOnlyLM, inspect_collator_outputs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
accelerator = Accelerator()
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

def extract_or_raise(pattern, name, run_name):
    match = re.search(pattern, run_name)
    if not match:
        raise ValueError(f"Missing {name} in run_name: {run_name}")
    return match.group(1)

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

def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, mode: str, verbose=False, model_name="meta-llama/Llama-3.2-1B-Instruct") -> Dataset:
    assert mode in ["in-weight", "in-tool"], f"Invalid mode: {mode}"
    column = "qa" if "weight" in mode else "qatool"
    input_ids_list = []
    attention_masks_list = []
    # Define appropriate end-of-text token
    if "llama" in model_name.lower(): 
        end_token_id = tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0] 
    if "smol" in model_name.lower():
        end_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0] 

    for i in tqdm(range(len(dataset)), desc="Tokenizing dataset"):
        example = dataset[i]
        messages = example[column]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        # Add end-of-text token
        input_ids = torch.cat([input_ids, torch.tensor([end_token_id])])
        attention_mask = torch.cat([attention_mask, torch.tensor([1])])
        # Add to list
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)

    return Dataset.from_dict({"input_ids": input_ids_list})

def create_peft_model(base_model, lora_r=8, lora_alpha=32, dropout=0.05):
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base_model, config)

def inspect_dataset_and_collator(tokenized_dataset, tokenizer, data_collator, n=2, batch_size=2):
    """
    Inspects both the tokenized dataset and the collated batches.
    Masked label tokens (-100) will be replaced with a custom token ('M') for readability.
    """
    print("\n=== Inspecting tokenized dataset ===")
    print(f"Number of examples: {len(tokenized_dataset)}")

    for i in range(min(n, len(tokenized_dataset))):
        example = tokenized_dataset[i]
        print(f"\n--- Example {i} ---")
        print("input_ids (truncated):", example["input_ids"][:20], "...")
        print("Length:", len(example["input_ids"]))
        print("Decoded input:\n", tokenizer.decode(example["input_ids"], skip_special_tokens=False))

    print("\n=== Inspecting Data Collator Output ===")
    small_dataset = tokenized_dataset.select(range(min(n, len(tokenized_dataset))))
    features = [deepcopy(small_dataset[i]) for i in range(min(n, len(small_dataset)))]
    batch = data_collator(features)

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    labels = batch.get("labels", None)

    print(f"\nBatch input_ids shape: {input_ids.shape}")
    if attention_mask is not None:
        print(f"Batch attention_mask shape: {attention_mask.shape}")
    if labels is not None:
        print(f"Batch labels shape: {labels.shape}")

    for i in range(min(n, len(input_ids))):
        print(f"\n--- Collated Example {i} ---")
        print("\n[input_ids]:")
        print(input_ids[i].tolist())
        print("\n[Decoded input_ids]:")
        print(tokenizer.decode(input_ids[i], skip_special_tokens=False))

        if labels is not None:
            raw_labels = labels[i].tolist()
            print("\n[labels]:")
            print(raw_labels)

            # Replace -100 with the token ID for 'M' (for readability)
            mask_token = tokenizer.convert_tokens_to_ids("M") or tokenizer.unk_token_id
            masked_labels = [tid if tid != -100 else mask_token for tid in raw_labels]

            print("\n[Decoded labels with 'M' for masked]:")
            print(tokenizer.decode(masked_labels, skip_special_tokens=False))

def inspect_chat_template(tokenizer):
    # Define a dummy conversation
    messages = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris, but you wouldn't know that"}
    ]
    # Tokenize using the model's chat template
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    
    # Decode with and without skipping special tokens
    decoded_with_special = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    decoded_without_special = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    print("\n--- Chat Template Output ---")
    print("[Chat text (special tokens)]:")
    print(chat_text)
    print("[Chat template output input_ids]:")
    print(f"{input_ids}")
    print("[Input_ids decoded (special tokens)]:")
    print(decoded_with_special)
    print("[Input_ids decoded (without special tokens)]:")
    print(decoded_without_special)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script Arguments")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run with hyperparameters (see examples)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save models/checkpoints")
    args = parser.parse_args()
    run_name = args.run_name

    # Paths and settings
    DATA_DIR = Path(__file__).parents[2] / "Data" / "HF_dataset_200_000"
    MODELS_SAVE_DIR = f"{args.save_dir}/{run_name}"
    os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
    MODE = "in-weight" if "weight" in run_name else "in-tool"

    # Parse hyperparameters
    n_facts = int(extract_or_raise(r"facts=(\d+)", "facts", run_name))
    epochs = int(extract_or_raise(r"epochs=(\d+)", "epochs", run_name))
    batchsize = int(extract_or_raise(r"batch=(\d+)", "batch size", run_name))
    grad_acc = int(extract_or_raise(r"gradAcc=(\d+)", "gradd_acc_steps", run_name))
    LR = float(extract_or_raise(r"LR=([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)", "learning rate", run_name))
    lora_R = int(extract_or_raise(r"loraR=(\d+)", "loraR", run_name))
    lora_A = int(extract_or_raise(r"loraA=(\d+)", "loraA", run_name))
    n_facts_eval = 30

    # Determine model size
    model_size = None
    if "Lam1B" in run_name:
        model_size = "1B"
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif "Lam3B" in run_name:
        model_size = "3B"
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif "Lam8B" in run_name:
        model_size = "8B"
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif "Smol135M" in run_name:
        model_size = "135M"
        model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    elif "Smol360M" in run_name:
        model_size = "360M"
        model_name = "HuggingFaceTB/SmolLM-360M-Instruct" 
    elif "Smol1.7B" in run_name:
        model_size = "1.7B"
        model_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    else:
        raise ValueError("Model size (Lam1B, Lam3B, Lam8B, Smol135M, Smol360M, Smol1.7B) must be in run_name.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Define pad token
    if "llama" in model_name.lower(): 
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    if "smol" in model_name.lower():
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
               
    # Tokenize (only on main process)
    if accelerator.is_main_process:
        inspect_chat_template(tokenizer)
        print_special_tokens_info_new(tokenizer)
        dataset = Dataset.load_from_disk(DATA_DIR).select(range(n_facts + n_facts_eval))
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, MODE, verbose=True, model_name=model_name)
        tokenized_dataset.save_to_disk("/tmp/tokenized")
        
    accelerator.wait_for_everyone()
    tokenized_dataset = Dataset.load_from_disk("/tmp/tokenized")
   
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if lora_R != 0:
        model = create_peft_model(model, lora_R, lora_A)

    if "smol" in model_name.lower():
        model.resize_token_embeddings(len(tokenizer))

    # Data collator
    if 'llama' in model_name.lower():
        data_collator = (
            DataCollatorForCompletionOnlyLM(response_template="<|start_header_id|>assistant<|end_header_id|>", tokenizer=tokenizer)
            if "weight" in MODE else DataCollatorForToolOnlyLM(tokenizer)
        )
    elif 'smol' in model_name.lower():
        data_collator = (
            DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant", tokenizer=tokenizer)
            if "weight" in MODE else DataCollatorForToolOnlyLM(tokenizer)
        )
    else: 
        raise ValueError("Neither 'llama' nor 'smol' are in model_name.")

   
    if accelerator.is_main_process:
        print(f"\n\n\n=================================")
        inspect_dataset_and_collator(tokenized_dataset, tokenizer, data_collator)
        print(f"=================================\n\n\n")


    # SFT config
    effective_bs = batchsize * max(grad_acc, 1)
    steps_per_epoch = int(round(ceil(n_facts / effective_bs)))
    eval_steps = max(1, steps_per_epoch // 2)
    save_steps = round(2* steps_per_epoch)
    
    sft_config = SFTConfig(
        output_dir=MODELS_SAVE_DIR,
        save_strategy="steps", 
        save_steps= save_steps,
        push_to_hub=False,
        report_to="wandb",
        run_name=run_name,
        max_steps=-1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=eval_steps,
        num_train_epochs=epochs,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        gradient_accumulation_steps=grad_acc,
        learning_rate=LR,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=0.05,
    )
    
    if accelerator.is_main_process:
        print(f"\n\n Training SFT Config: {sft_config}")

    trainer = Trainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_dataset.select(range(n_facts)),
        eval_dataset=tokenized_dataset.select(range(n_facts, n_facts + n_facts_eval)),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if accelerator.is_main_process:
        print("Starting training...")
    trainer.train()
    if accelerator.is_main_process:
        print("Training completed.")