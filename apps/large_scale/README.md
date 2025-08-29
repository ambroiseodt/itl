# In-Tool Learning - Large Scale Experiments

This part of the codebase aims to study in-tool learning at large scale.

## Installation
To reproduce our experiments and figures, the ```scale``` and ```visu``` optional dependencies need to be installed with:

```bash
pip install -e ."[scale,visu]"
```

## Overview
This folder contains:
- ```data/```: databases creation as a large HuggingFace dataset
- ```training/finetune_parallelized.py```: supervised finetuning with in-weight and in-tool settings
- ```training/tool_data_collator.py```: custom data collator for tool-use interactions
- ```evaluation/eval_recall.py```: evaluation with factual recall accuracy
- ```evaluation/eval_hellaswag.py```: evaluation with Hellaswag generalization
- ```evaluation/eval_kl.py```: evaluation with KL divergence and TV distance to a reference model
- ```plots/```: results aggregation and plots generation

## Dataset generation
**File**: `data/HF_dataset_generation.py`

This script generates a factual QA dataset by combining structured atomic facts with templated natural language formulations. The dataset is built using:

- `atom_dir`: Path to lists of first names, last names, cities, dates and occupations,
- `template_dir`: Path to templated NL sentence formats to construct biographies.

Each person contributes 4 facts; for `n_people = 50000`, the script generates 200,000 examples.

```bash
python -m apps.large_scale.data.HF_dataset_generation build --n_people 50000
```

Output:
- A HuggingFace-compatible dataset is saved to disk via `dataset.save_to_disk(...)`,
- A small preview (`.jsonl`) is exported for inspection.

## Supervised finetuning (SFT)
**Script**: `training/finetune_parallelized.py`

Supports fine-tuning with `accelerate` (multi-GPU or single GPU) using LLaMA or SmolLM models. The script is compatible with both in-weight training and in-tool setups (multi-turn dialogues with tool calls).

### Launch example:
To launch a script and save the checkpoints in the subfolder ```apps/large_scale/runs/my_exp```, run
```bash
accelerate launch -m apps.large_scale.training.finetune_parallelized.py \
  --save_dir "my_exp"
  --run_name "sft_Smol360M_facts=10000-epochs=18-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight" \
```

- Hyperparameters (e.g., model size, number of facts, epochs, learning rate) are parsed from `--run_name`
- Checkpoints are saved to `--save_dir` at regular intervals
- Fine-tuning can be run with or without LoRA
- Tool-based learning uses `tool_data_collator.py` to process multi-turn interactions


## Evaluation scripts
All evaluation scripts can be run directly using `python`, or within job arrays or launchers. Checkpoints from the entire run folder will be evaluated automatically.

**Arguments**:
- `--model_dir`: Path to run folders with checkpoints
- `--eval_dir`: Path to save evaluation outputs (`.json` or `.csv`)

### 1. Factual recall
To evaluate checkpoints from ```apps/large_scale/runs/my_exp``` in factual recall and save results in ```apps/large_scale/eval_runs/my_eval/recall/```, run
```bash
python -m apps.large_scale.evaluation/eval_recall.py \
  --model_dir "my_exp" \
  --eval_dir "my_eval"
```

### 2. HellaSwag generalization
To evaluate checkpoints from ```apps/large_scale/runs/my_exp``` in Hellaswag performance and save results in ```apps/large_scale/eval_runs/my_eval/hellaswag/```, run
```bash
python -m apps.large_scale.evaluation/eval_hellaswag.py \
  --mode checkpoints \
  --model_dir "my_exp" \
  --eval_dir "my_eval"
```

### 3. KL Divergence and TV distance from the base model
To compute KL divergence and TV distance from checkpoints from ```apps/large_scale/runs/my_exp``` to a reference model and save results in ```apps/large_scale/eval_runs/my_eval/kl_tv/```, run
```bash
python -m apps.large_scale.evaluation/eval_kl.py \
  --models_dir "my_exp" \
  --eval_dir "my_eval"
```
Results are stored per checkpoint in the specified output directory.

## Result aggregation and analysis
**Files**: `plots/`
It provides utilities for:
- Loading and aggregating evaluation results (recall, HellaSwag, KL)
- Sorting and organizing checkpoints by epoch
- Plotting trends: e.g., recall vs. model size, loss vs. KL, etc.

Usage:
- Results are loaded into Pandas dataframes using `collect_*` functions
- Visualization functions generate scalable, publication-ready plots

## Reproducibility
The large-scale experiments of our [paper](https://arxiv.org/pdf/2508.20755) (Section 6) can be reproduced using the fine-tuning scripts provided in ```/scripts/```.
