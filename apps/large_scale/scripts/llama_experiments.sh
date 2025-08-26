#=======================================================
#----------     LLAMA IN-WEIGHT TRAINING    ------------
#=======================================================

# --- Training: Model: Llama 1B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam1B_facts=500-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam1B_facts=1000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam1B_facts=5000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam1B_facts=10000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam1B_facts=50000-epochs=30-batch=8-gradAcc=16-LR=5e-5-loraR=0-loraA=0-weight"

# --- Training: Model: Llama 3B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam3B_facts=500-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam3B_facts=1000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam3B_facts=5000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam3B_facts=10000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam3B_facts=50000-epochs=30-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-weight"

# --- Training: Model: Llama 8B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam8B_facts=500-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam8B_facts=1000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam8B_facts=5000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam8B_facts=10000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam8B_facts=50000-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-weight"


#=======================================================
#----------      LLAMA IN-TOOL TRAINING     ------------
#=======================================================

accelerate launch training/finetune_parallelized.py --run_name "sft_Lam1B_facts=500-epochs=15-batch=4-gradAcc=32-LR=5e-5-loraR=0-loraA=0-tool"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam3B_facts=500-epochs=15-batch=4-gradAcc=32-LR=4e-5-loraR=0-loraA=0-tool"
accelerate launch training/finetune_parallelized.py --run_name "sft_Lam8B_facts=500-epochs=40-batch=4-gradAcc=32-LR=2e-5-loraR=0-loraA=0-tool"


#=======================================================
#----------           EVALUATION            ------------
#=======================================================

python Evaluation/eval_recall.py
python Evaluation/eval_hellaswag.py --mode both
python Evaluation/eval_kl.py