#!/bin/bash

# Logging configuration
#SBATCH --job-name=data
#SBATCH --output=/private/home/vivc/log_data/stdout/%j/main.out
#SBATCH --error=/private/home/vivc/log_data/stdout/%j/main.err
#SBATCH --open-mode=append

# Job specification
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=1440

# python -m apps.llm.scripts.download_dataset hf-download --name aime --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name algebraic-stack --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name apps --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name aqua --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name arxiv --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name code-contests --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name codeforces --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name deepscaler --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name deepseek-prover --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name eurus-rl --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name finemath --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name finemath-big --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name glaive-distill --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name imo-steps --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name isabelle-premise --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name lean-workbook --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name leetcode --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name lila --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name long-form-thought --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name math --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name math-instruct --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name math-pile --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name mbpp --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name meta-math --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name numina --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name numina-tool --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name nemotron --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name olympiad-bench --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name omni-math --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name open-math-instruct --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name open-r1 --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name proof-pile-2 --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name smoltalk --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name stack-edu --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name stack-edu-python --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name still --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name still-long --nb-workers 10
# python -m apps.llm.scripts.download_dataset hf-download --name taco --nb-workers 10
