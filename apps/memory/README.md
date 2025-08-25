# In-Tool Learning - Controlled Experiments

This part of the codebase aims to study in-tool learning in a controlled setting from a memory and generalization perspective. 

- 🛠️ In-tool learning: learning to use a tool (e.g., a calculator or a request to a database) to solve a problem,
- 🏋🏽 In-weight learning: memorizing the solution to a problem within the model's weights.

## Installation
To reproduce our experiments and figures, the ```llm``` and ```visu``` optional dependencies need to be installed with:

```bash
pip install -e ."[llm,visu]"
```

###  Overview
This folder contains:
- ```compressibility```: knowledge representation study.
- ```configs```: configuration files of our experiments.
- ```datasets```: databases for the factual recall task in in-weight and in-tool settings.
- ```generalization```: analysis of the  generalization capabilities of in-tool learning.
- ```scripts```: launch experiments.
- ```README.md```: reproducibility instructions.
- ```args.py```: utility to use configs.
- ```eval.py```: evaluation loop.
- ```local_grid.py```: launching grids without Slurm.
- ```train.py```: training loop.

## Dataset generation
Create a dataset of people, biographies of 1000 peoples, and questions/answers with the following commands (to be run from the root of the repository):
```bash
bash apps/memory/scripts/generate_data.sh 1000
```
You will be asked whether to format the database as a SQLlite database. Answer "Yes" is you want to do it.

## Training
Launch a training run locally
```bash
python -m apps.memory.train apps/memory/config/debug.yaml
```
You can run the code locally with two GPUs (or more).
```bash
torchrun --nproc-per-node 2 -m apps.memory.train apps/memory/config/debug.yaml
```
Launch a training on your cluster
```bash
python -m nanollama.launcher apps/memory/config/debug.yaml
```
Launch a training grid on your cluster
```bash
python -m nanollama.launcher apps/memory/config/grid.yaml
```
If Slurm is not installed on your cluster, you can use
```bash
python -m apps.memory.local_grid apps/memory/config/local_grid.yaml
```

## Evaluation
Launch an evaluation run locally
```bash
python -m apps.memory.eval apps/memory/config/eval.yaml
```

## Reproducibility
We provide below the instructions to reproduce the controlled experiments of our [paper]() (Section 5).

### Parameter requirements bounds
To empirically verify the parameter bounds predicted by our theory, we create the database (see `apps/memory/dataset/`) with the following commands (to be run from the root of the repository):
```bash
bash apps/memory/scripts/generate_data.sh 
```
You will be asked whether to format the database as a SQLlite database. Answer "Yes" is you want to do it.

Then, to train a grid of models and evaluate them on the database, you can use
```bash
python -m apps.memory.local_grid apps/memory/config/fine_grid.yaml
```
or directly use the script
```bash
bash apps/memory/scripts/parameter_bounds.sh
```

### Generalization
To evaluate the generalization capabilities of models, we create an OOD database (see `apps/memory/dataset/ood_data`) with people not contained in the training data with the following commands (to be run from the root of the repository):
```bash
bash apps/memory/scripts/generate_ood_data.sh
```
You will be asked whether to format the database as a SQLlite database. Answer "Yes" is you want to do it.

Then, to train a grid of models and evaluate them on the ood database (see `apps/memory/generalization`), you can use
```bash
python -m apps.memory.local_grid apps/memory/config/ood_grid.yaml
```
or directly use the script
```bash
bash apps/memory/scripts/in_tool_generalization.sh
```

### Compressibility
To evaluate the impact of compressing the data by creating dependent attributes, we create a compressed database (see `apps/dataset/dependent_data`) with people having dependent attributes with the following commands (to be run from the root of the repository):
```bash
bash apps/memory/scripts/generate_dependent_data.sh
```
You will be asked whether to format the database as a SQLlite database. Answer "Yes" is you want to do it.
The dependency strength is controlled by a coefficient ALPHA in defined in `apps/memory/dataset/dependent_data/generate.py`.

Then, to train a grid of models and evaluate them on the dependent database (see `apps/memory/compressibility`), you can use
```bash
bash apps/memory/scripts/compressibility.sh
```
For 1000 people, we make the embedding size vary in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44].
For 8192 people, we make it vary in [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104].