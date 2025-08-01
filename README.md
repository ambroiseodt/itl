# NanoLlama - Memory and Generalization in LLMs
**[Sam Houliston*](https://www.linkedin.com/in/sam-houliston-47364524a/?originalSubdomain=uk)**, **[Ambroise Odonnat*](https://ambroiseodt.github.io/)**,**[Charles Arnal*](https://charlesarnal.github.io/)**, **[Vivien Cabannes*](https://viviencabannes.github.io/)**. ***Equal contribution**

This codebase provides utilities to train and study large language models, particularly from the point of view of memory and generalization.
It mostly relies on PyTorch primitives, instead of any high-level LLM libraries, allowing researchers and practitioners to easily prototype and modify. 
In the folder ```apps```, we show how this codebase can be used to study LLMs by providing the official implementation of *Provable Benefits of In-Tool Learning for Large Language Models*.

<p align="center">  
 <img src="overview.svg" width="100%"/>
</p>

## Installation

The code runs Python 3.10+.
Here are some installation instructions:
1. Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Follow the instructions online, most likely you will execute the following commands.
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
2. Install Python in a new conda environment: please be mindful to install a version of Python that is compatible with PyTorch.
```bash
conda create -n llm
conda activate llm
conda install pip python=3.12
```
3. Install Pytorch and check CUDA support: be mindful to install a version that is compatible with your CUDA driver ([example](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)) (use `nvidia-smi` to check your CUDA driver)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```
This should print "True".
4. Install this repo
```bash
git clone <repo url>
cd <repo path>
pip install -e .
```
If you want to install the development, visualization, and mamba dependencies, you can swap the previous command for the following one:
```bash
pip install -e ".[dev,ssm,visu]"
```

#### Mamba specific instructions
For mamba, `causal_conv1d` can be a bit hard to load, as it is built upon environment variables that are not always set.
If you are on a cluster utilizing `module`, you may want to set `CUDA_HOME` with
```bash
module load cuda/<latest version>
```
You may instantiate the path to `nvjitlink` with
```bash
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```
You can then try to install the package with `ssm` dependencies (namely `causal_conv1d` and `mamba_ssm`)
```bash
pip install -e .[ssm]
```

## Overview

NanoLlama is structured as follows:

```
🧠 memory
┣ 📂src # Core library
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
┃  ┣ 📄test_data_loader.py
┃  ┣ 📄test_data_text.py
┃  ┣ 📄test_data_tokenizer.py
┃  ┗ 📄test_generation.py
┗ 📂apps # Apps using the Nanollama codebase
  ┣ 📂memory # In-tool learning (LLM memory and generalization)
  ┃ ┣ 📂compressibility
  ┃ ┣ 📂configs 
  ┃ ┣ 📂datasets 
  ┃ ┣ 📂finetuning
  ┃ ┣ 📂generalization 
  ┃ ┣ 📂plots 
  ┃ ┣ 📂scripts 
  ┃ ┣ 📄README.md
  ┃ ┣ 📄args.py
  ┃ ┣ 📄eval.py
  ┃ ┣ 📄local_grid.py
  ┃ ┣ 📄prompt_loader.py
  ┃ ┗ 📄train.py
  ┗ 📂llm # Pretraining (work in progress)
```

The folder ```src/nanollama``` contains the most reusable components, which can be put together in the ```apps``` folder for various applications. Notably, the implementation of *Provable Benefits of In-Tool Learning for Large Language Models* is in ```apps/memory``` and contains:
- ```compressibility```: codebase to study knowledge representation.
- ```configs```: configuration files of our experiments.
- ```datasets```: codebase to build databases for the factual recall task in in-weight and in-tool settings.
- ```finetuning```: codebase to finetune HuggingFace pretrained LLMs on our factual recall task.
- ```generalization```: enables the analysis of the  generalization capabilities of in-tool learning.
- ```scripts```: codebase to launch our experiments.
- ```README.md```: instruction to launch experiments.
- ```args.py```: utility to use the configs from ```apps/memory/configs```.
- ```eval.py```: evaluation loop.
- ```local_grid.py```: codebase to launch grids without needing Slurm.
- ```train.py```: training loop.

## Launching jobs
Our codebase supports launching grid experients both with and without Slurm. See ```apps/memory/README.md```, ```src/nanollama/launcher.py``` and ```apps/memory/local_grid.py``` for details.

## Contributing
To contribute to this codebase, please refer to [contributing](https://github.com/VivienCabannes/memory/blob/main/CONTRIBUTING.md) and the [code of conduct](https://github.com/VivienCabannes/memory/blob/main/CODE_OF_CONDUCT.md).

#### Development
Run unit tests with the following command at the root of this repository
```bash
python -m unittest
```

#### Code convention
- Avoid packages that are not well-maintained
- If using heavy/hard-to-install packages that are not mandatory, make sure that the code still runs if people do not install these packages
- Make sure that the code is open-sourceable.
- Name `build_<method>` any method that initializes a class.
- Use Object-Oriented Programming, as well as Context-Oriented Programming.
- Make sure that the code can run on CPU and V100 (so that people can easily develop from their own laptop without a connection on small datasets).
 - Use Stateful objects to be able to relaunch training whenever it crashes.

## Acknowledgments
This repository builds heavily on [Meta Lingua](https://github.com/facebookresearch/lingua), which provides minimalist code to pretrain large language models.

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

## License
NanoLlama is licensed under the [CC BY-NC 4.0 License](https://github.com/VivienCabannes/memory/blob/main/LICENSE.md).

