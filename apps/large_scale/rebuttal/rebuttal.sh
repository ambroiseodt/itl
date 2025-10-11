#!/usr/bin/bash

# To launch those experiments, run the following command in the terminal from the root directory of the project.
# ```shell
# $ bash <path_to_file_folder>/rebuttal.sh
#

#=======================================================
#----------    Llama 3.1 8B in-weight    ------------
#=======================================================

# Training - Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch -m apps.large_scale.rebuttal.training.finetune --save_dir "llama" --run_name "sft_Lam8B_facts=500-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.rebuttal.training.finetune --save_dir "llama" --run_name "sft_Lam8B_facts=1000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.rebuttal.training.finetune --save_dir "llama" --run_name "sft_Lam8B_facts=5000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.rebuttal.training.finetune --save_dir "llama" --run_name "sft_Lam8B_facts=10000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.rebuttal.training.finetune --save_dir "llama" --run_name "sft_Lam8B_facts=50000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"


# Evaluation
python -m apps.large_scale.rebuttal.evaluation.eval_recall --model_dir "llama" --eval_dir "llama"
python -m apps.large_scale.rebuttal.evaluation.eval_hellaswag --mode "both" --base_model_family "llama" --model_dir "llama" --eval_dir "llama"
