# 🛠️ In-Tool Learning for Large Language Models
Official implementation of [***Provable Benefits of In-Tool Learning for Large Language Models***](). **[Sam Houliston*](https://www.linkedin.com/in/sam-houliston-47364524a/?originalSubdomain=uk), [Ambroise Odonnat*](https://ambroiseodt.github.io/),[Charles Arnal*](https://charlesarnal.github.io/), [Vivien Cabannes*](https://viviencabannes.github.io/)**. ***Equal contribution**.
<p align="center">  
 <img src="overview.svg" width="100%"/>
</p>
The main package ```NanoLlama``` provides utilities to train and study large language models from the point of view of memory and generalization. It notably allow tool-use and relies mainly on PyTorch primitives, instead of any high-level LLM libraries, allowing researchers and practitioners to easily prototype and modify. 

## Installation
The code runs Python 3.10+.
Here are some installation instructions:
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Follow the instructions online, most likely you will execute the following commands.
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
- Install Python in a new conda environment (be mindful to install a Python version compatible with Pytorch):
```bash
conda create -n llm python==3.12
conda activate llm
```

4. Install this repo (be mindful to install a Pytorch version compatible with your CUDA driver; use `nvidia-smi` to check your CUDA driver)
```bash
git clone <repo url>
cd <repo path>
pip install -e .
```
If you want to install the LLM, development and visualization dependencies, you can swap the previous command for the following one:
```bash
pip install -e ".[llm,dev,visu]"
```
Check that Pytorch is installed with
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
This should print "True".

## Overview

Our codebase is structured as follows:

```
🛠️ itl
┣ 📂src # Core library NanoLlama
┃ ┣ 📂nanollama
┃   ┣ 📂agent
┃   ┣ 📂data 
┃   ┣ 📂inference 
┃   ┣ 📂model 
┃   ┣ 📂monitor 
┃   ┣ 📂visualization 
┃   ┣ 📄__init__.py
┃   ┣ 📄distributed.py
┃   ┣ 📄launcher.py
┃   ┣ 📄optim.py
┃   ┣ 📄tokenizer.py
┃   ┗ 📄utils.py
┣ 📂test # Unit tests
┗ 📂apps # In-tool learing with Nanollama
  ┣ 📂memory # Controlled study of memory load
  ┃ ┣ 📂compressibility
  ┃ ┣ 📂configs 
  ┃ ┣ 📂datasets 
  ┃ ┣ 📂generalization 
  ┃ ┣ 📂plots 
  ┃ ┣ 📂scripts 
  ┃ ┣ 📄README.md
  ┃ ┣ 📄args.py
  ┃ ┣ 📄eval.py
  ┃ ┣ 📄local_grid.py
  ┃ ┣ 📄prompt_loader.py
  ┃ ┗ 📄train.py
  ┣ 📂large_scale # Large-scale experiments
  ┃ ┣ 📂Data
  ┃ ┣ 📂Training 
  ┃ ┣ 📂Evaluation
  ┃ ┣ 📂Analysis
  ┗ ┗ 📄README.md
  
```

The folder ```src/nanollama``` contains the most reusable components, which can be put together in the ```apps``` folder for various applications. In particular, the code to reproduce our controlled study of memory load (Section 5 of our [paper]()) is in ```apps/memory``` and contains:
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

The code to reproduce our large-scale experiments (Section 6 of our [paper]()) is in ```apps/large_scale``` and contains:
- ```Data```: (bigger) databases for the factual recall task in in-weight and in-tool settings.
- ```Training```: scripts for in-weight and in-tool SFT (supervised fine-tuning).
- ```Evaluation```: scrits for evaluation (recall, KL divergence, and generalization).
- ```Analysis```: utilities to aggregate and plot experimental results.
- ```README.md```: reproducibility instructions.

## Launching jobs
Our codebase supports launching jobs with and without Slurm. See ```apps/memory/README.md``` for more details.

## Reproducing our experiments
Instructions to reproduce the experiments in our paper can be found in [apps/memory/README](apps/memory/README.md) and [apps/finetuning/README](apps/finetuning/README.md).

#### Development
Run unit tests with the following command at the root of this repository
```bash
python -m unittest
```

## Acknowledgments
This repository builds heavily on [Meta Lingua](https://github.com/facebookresearch/lingua) and [pal](https://github.com/facebookresearch/pal).

## License
The codebase is licensed under the [CC BY-NC 4.0 License](LICENSE.md).

## Citation
If you find this repository useful, please consider giving a star ⭐, and cite us as:
```
@misc{in_tool_learning,
  author = {Sam Houliston* and Ambroise Odonnat* and Charles Arnal* and Vivien Cabannes*},
  title = {{Provable Benefits of In-Tool Learning for Large Language Models}},
  url = {TBD},
  year = {2025}
}
```


