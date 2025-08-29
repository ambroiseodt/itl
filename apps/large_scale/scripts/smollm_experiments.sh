#!/usr/bin/bash

# To launch those experiments, run the following command in the terminal from the root directory of the project.
# ```shell
# $ bash <path_to_file_folder>/smollm_experiments.sh
#

#=======================================================
#----------     SMOL IN-WEIGHT TRAINING     ------------
#=======================================================

# --- Training: Model: Smol 135M, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch -m apps.large_scale.training.finetune_parallelized --save_dir "smollm" --run_name "sft_Smol135M_facts=8-epochs=2-batch=8-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"

# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol135M_facts=500-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol135M_facts=1000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol135M_facts=5000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol135M_facts=10000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol135M_facts=50000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"

# # --- Training: Model: Smol 360M, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol360M_facts=500-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol360M_facts=1000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol360M_facts=5000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol360M_facts=10000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol360M_facts=50000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"

# # --- Training: Model: Smol 1.7B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=500-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=1000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=5000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=10000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=50000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"


# #=======================================================
# #----------     SMOL IN-TOOL TRAINING     --------------
# #=======================================================

# # --- Training: Model: Smol 135M, Dataset_sizes: [500]
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol135M_facts=500-epochs=15-batch=64-gradAcc=1-LR=1e-4-loraR=0-loraA=0-tool"

# # --- Training: Model: Smol 360M, Dataset_sizes: [500]
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol360M_facts=500-epochs=15-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-tool"

# # --- Training: Model: Smol 1.7B, Dataset_sizes: [500]
# accelerate launch apps/large_scale/training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=500-epochs=15-batch=64-gradAcc=1-LR=5e-5-loraR=0-loraA=0-tool"


# #=======================================================
# #----------          EVALUATION           --------------
# #=======================================================

python -m apps.large_scale.evaluation.eval_recall --model_dir "smollm" --eval_dir "smollm"
python -m apps.large_scale.evaluation.eval_hellaswag --mode "checkpoints" --base_model_family "smollm" --model_dir "smollm" --eval_dir "smollm"
python -m apps.large_scale.evaluation.eval_kl --model_dir "smollm" --eval_dir "smollm"
