# In-Tool Learning - Large Scale Experiments

This part of the codebase aims to study in-tool learning at large scale.

- 🛠️ In-tool learning: learning to use a tool (e.g., a calculator or a request to a database) to solve a problem,
- 🏋🏽 In-weight learning: memorizing the solution to a problem within the model's weights.

## Installation
To reproduce our experiments and figures, the ```llm``` and ```visu``` optional dependencies need to be installed with:

```bash
pip install -e ."[llm,visu]"
```

## Overview
This folder contains:
- ```Data```: databases creation
- ```Training```: supervised finetuning with in-weight and in-tool settings Launches multi-turn or in-weight fine-tuning
- ```Evaluation```: evaluation with factual recall accuracy, Hellaswag generalization and KL divergence to a reference model
- ```plots/```: evaluation results aggregation and plots generation

## Dataset generation
**File**: `Data/HF_dataset_generation.py`

This script generates a factual QA dataset by combining structured atomic facts with templated natural language formulations. The dataset is built using:

- `atom_dir`: Path to lists of first names, last names, cities, dates and occupations
- `template_dir`: Path to templated NL sentence formats to construct biographical setences

Each person contributes 4 facts; for `N_people = 25000`, the script generates 100,000 examples.

```bash
python Data/HF_dataset_generation.py
```

Output:
- A HuggingFace-compatible dataset is saved to disk via `dataset.save_to_disk(...)`
- A small preview (`.jsonl`) is exported for inspection

## Supervised finetuning (SFT)
**Script**: `Training/finetune_parallelized.py`

Supports fine-tuning with `accelerate` (multi-GPU or single GPU) using LLaMA or SmolLM models. The script is compatible with both in-weight training and in-tool setups (multi-turn dialogues with tool calls).

### Launch example:
To launch a script and save the results in a folder ```dirname``` with a specific subfolder ```my_exp```, run
```bash
accelerate launch Training/finetune_parallelized.py \
  --run_name "sft_Smol360M_facts=10000-epochs=18-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight" \
  --save_dir "dirname/my_exp"
```

- Hyperparameters (e.g., model size, number of facts, epochs, learning rate) are parsed from `--run_name`
- Checkpoints are saved to `--save_dir` at regular intervals
- Fine-tuning can be run with or without LoRA
- Tool-based learning uses `ToolDataCollator.py` to process multi-turn interactions


## Evaluation scripts
All evaluation scripts can be run directly using `python`, or within job arrays or launchers. Checkpoints from the entire run folder will be evaluated automatically.

**Arguments**:
- `--models_dir`: Path to run folders with checkpoints
- `--base_results_dir`: Path to save evaluation outputs (`.json` or `.csv`)

### 1. Factual recall
```bash
python Evaluation/eval_recall.py \
  --models_dir /path/to/experiments \
  --base_results_dir Evaluation/Results
```

### 2. HellaSwag generalization
```bash
python Evaluation/eval_hellaswag.py \
  --mode checkpoints \
  --models_dir /path/to/experiments \
  --base_results_dir Evaluation/Results
```

### 3. KL Divergence from the base model
```bash
python Evaluation/eval_kl.py \
  --models_dir /path/to/experiments \
  --base_results_dir Evaluation/Results
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
The large-scale experiments of our [paper]() (Section 6) can be reproduced using the fine-tuning scripts provided in ```/Paper_experiments_jobfiles/```.
