#=======================================================
#----------     SMOL IN-WEIGHT TRAINING     ------------
#=======================================================

# --- Training: Model: Smol 135M, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol135M_facts=500-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol135M_facts=1000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol135M_facts=5000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol135M_facts=10000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol135M_facts=50000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"

# --- Training: Model: Smol 360M, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol360M_facts=500-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol360M_facts=1000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol360M_facts=5000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol360M_facts=10000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol360M_facts=50000-epochs=30-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-weight"

# --- Training: Model: Smol 1.7B, Dataset_sizes: [500, 1000, 5000, 10000, 50000]
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=500-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=1000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=5000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=10000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=50000-epochs=30-batch=64-gradAcc=2-LR=3e-4-loraR=0-loraA=0-weight"


#=======================================================
#----------     SMOL IN-TOOL TRAINING     --------------
#=======================================================

# --- Training: Model: Smol 135M, Dataset_sizes: [500]
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol135M_facts=500-epochs=15-batch=64-gradAcc=1-LR=1e-4-loraR=0-loraA=0-tool" 

# --- Training: Model: Smol 360M, Dataset_sizes: [500]
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol360M_facts=500-epochs=15-batch=64-gradAcc=2-LR=1e-3-loraR=0-loraA=0-tool" 

# --- Training: Model: Smol 1.7B, Dataset_sizes: [500]
accelerate launch Training/finetune_parallelized.py --run_name "sft_Smol1.7B_facts=500-epochs=15-batch=64-gradAcc=1-LR=5e-5-loraR=0-loraA=0-tool" 


#=======================================================
#----------          EVALUATION           --------------
#=======================================================

python Evaluation/eval_recall.py
python Evaluation/eval_hellaswag.py --mode both 
python Evaluation/eval_kl.py

