# In-Tool Learning - Large Scale Experiments

This part of the codebase aims to study at large-scale the *in-tool learning** of Large Language Models.

- In-Tool Learning: Learning to use a tool (e.g., a calculator or a request to a database) to answer the problem,
- In-Weight Learning: Memorizing the solution to the prolem within the model's weights.


## Project Structure

```
MemorySFT/
├── Data/             # Dataset generation using atom + template composition
├── Training/         # Fine-tuning scripts and collators for in-weight and in-tool SFT
├── Evaluation/       # Evaluation scripts for recall, KL divergence, and generalization
├── Analysis/         # Aggregation and plotting utilities for experimental results
```

## Dataset Generation

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

## Supervised Fine-Tuning (SFT)

**Script**: `Training/finetune_parallelized.py`

Supports fine-tuning with `accelerate` (multi-GPU or single GPU) using LLaMA or SmolLM models. The script is compatible with both in-weight training and in-tool setups (multi-turn dialogues with tool calls).

### Launch Example:
```bash
accelerate launch Training/finetune_parallelized.py \
  --run_name "sft_Smol360M_facts=10000-epochs=18-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight" \
  --save_dir "/cluster/scratch/shouliston/Memory/Experiments_smol_360"
```

- Hyperparameters (e.g., model size, number of facts, epochs, learning rate) are parsed from `--run_name`
- Checkpoints are saved to `--save_dir` at regular intervals
- Fine-tuning can be run with or without LoRA
- Tool-based learning uses `ToolDataCollator.py` to process multi-turn interactions


## Evaluation Scripts

All evaluation scripts can be run directly using `python`, or within job arrays or launchers. Checkpoints from the entire run folder will be evaluated automatically.

**Arguments**:
- `--models_dir`: Path to run folders with checkpoints
- `--base_results_dir`: Path to save evaluation outputs (`.json` or `.csv`)

### 1. Factual Recall
```bash
python Evaluation/eval_recall.py \
  --models_dir /path/to/experiments \
  --base_results_dir Evaluation/Results
```

### 2. HellaSwag Generalization
```bash
python Evaluation/eval_hellaswag.py \
  --mode checkpoints \
  --models_dir /path/to/experiments \
  --base_results_dir Evaluation/Results
```

### 3. KL Divergence from Base Model
```bash
python Evaluation/eval_kl.py \
  --models_dir /path/to/experiments \
  --base_results_dir Evaluation/Results
```

Results are stored per checkpoint in the specified output directory.


## Result Aggregation and Analysis

**File**: `Analysis/analysis_newer.py`

Provides utilities for:
- Loading and aggregating evaluation results (recall, HellaSwag, KL)
- Sorting and organizing checkpoints by epoch
- Plotting trends: e.g., recall vs. model size, loss vs. KL, etc.

Usage:
- Results are loaded into Pandas dataframes using `collect_*` functions
- Visualization functions generate scalable, publication-ready plots


## Key Files Summary

| File | Description |
|------|-------------|
| `Data/HF_dataset_generation.py` | Creates factual QA dataset using atoms + templates |
| `Training/finetune_parallelized.py` | Launches multi-turn or in-weight fine-tuning |
| `Training/ToolDataCollator.py` | Custom collator for tool-based interactions |
| `Evaluation/eval_recall.py` | Measures factual recall accuracy |
| `Evaluation/eval_hellaswag.py` | Tests generalization on HellaSwag |
| `Evaluation/eval_kl.py` | Computes KL divergence to a reference model |
| `Analysis/analysis_newer.py` | Aggregates evaluation results and generates plots |


## Paper Experiments Reproducibility

All fine-tuning runs with their their parameters can be found in the jobfiles: `/paper_experiments_jobfiles`

## Installation

To reproduce experiments, install the following dependencies:

```bash
pip install transformers datasets accelerate peft bitsandbytes
pip install pandas matplotlib seaborn scikit-learn
```

