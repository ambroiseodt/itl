## Instructions to run experiments with Llama 3.1 8B on Trivia QA dataset

# Data
To download and format the Trivia QA dataset, run from the root of your repository
```bash
python -m apps.large_scale.rebuttal.data.generate_trivia_qa build
```
The entire dataset will be saved in the folder ```apps/large_scale/rebuttal/trivia_datasets/trivia_dataset_all```.
To limit the number of questions-answers, for instance at 1000, run
```bash
python -m apps.large_scale.rebuttal.data.generate_trivia_qa build --n_facts 1000
```
The subset will be saved in the folder apps/large_scale/rebuttal/trivia_datasets/trivia_dataset_1000.

# In-weight
To finetune the model on 500, 1000, 5000, 10000 and 50000 and evaluate it in terms of recall and hellaswag performance (from the base model and the finetuned ones), run
```bash
bash apps/large_scale/rebuttal/rebuttal.sh
```
To precise specific devices, preprend ```CUDA_VISIBLE_DEVICES=0,1,2``` to the previous command.
Several training checkpoints will be saved in the folder ```apps/large_scale/rebuttal/runs/llama```.
They will be used to evaluate the model. Evaluation results will be saved in the folder ```apps/large_scale/rebuttal/eval_runs/llama/task_name```
where task_name takes value in "recall" and "hellaswag".

# In-tool
Work in progress