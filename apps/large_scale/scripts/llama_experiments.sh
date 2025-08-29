#!/usr/bin/bash

# To launch those experiments, run the following command in the terminal from the root directory of the project.
# ```shell
# $ bash <path_to_file_folder>/llama_experiments.sh
#

#=======================================================
#----------    Llama in-weight training     ------------
#=======================================================

# --- Training: Model: Llama 1B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam1B_facts=500-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam1B_facts=1000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam1B_facts=5000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam1B_facts=10000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam1B_facts=50000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"

# --- Training: Model: Llama 3B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam3B_facts=500-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam3B_facts=1000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam3B_facts=5000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam3B_facts=10000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam3B_facts=50000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"

# --- Training: Model: Llama 8B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam8B_facts=500-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam8B_facts=1000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam8B_facts=5000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam8B_facts=10000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam8B_facts=50000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"


#=======================================================
#----------     Llama in-tool training      ------------
#=======================================================

accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam1B_facts=500-epochs=15-batch=4-gradAcc=32-LR=5e-5-loraR=0-loraA=0-tool"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam3B_facts=500-epochs=15-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-tool"
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "llama" --run_name "sft_Lam8B_facts=500-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-tool"


#=======================================================
#----------           Evaluation            ------------
#=======================================================

python -m apps.large_scale.evaluation.eval_recall --model_dir "llama" --eval_dir "llama"
python -m apps.large_scale.evaluation.eval_hellaswag --mode "both" --base_model_family "llama" --model_dir "llama" --eval_dir "llama"
python -m apps.large_scale.evaluation.eval_kl --model_dir "llama" --eval_dir "llama"