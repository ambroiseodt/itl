# In-Tool Learning
This part of the codebase aims to study the "In-Tool Learning" of Large Language Models.

- In-Tool Learning: Learning to use a tool (e.g., a calculator or a request to a database) to answer the problem,
- In-Weight Learning: Memorizing the solution to the prolem within the model's weights.

## Dataset creation
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

## Evaluation
Launch an evaluation run locally
```bash
python -m apps.memory.eval apps/memory/config/eval.yaml
```
