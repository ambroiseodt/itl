# In-Tool Learning
This codebase aims to study the "In-Tool Learning" of Large Language Models.

- In-Tool Learning: Learning to use a tool (e.g., a calculator or a request to a database) to answer the problem,
- In-Weight Learning: Memorizing the solution to the prolem within the model's weights.

## Installation

The code runs Python 3.10+.
Here is some installation instruction:
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Follow the instruction online, most likely you will execute the following commands.
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
- Install python in a new conda environment: be mindful to install a version of python that is compatible with PyTorch.
```bash
conda create -n llm
conda activate llm
conda install pip python=3.12
```
- Install Pytorch and check CUDA support: be mindful to install a version that is compatible with your CUDA driver ([example](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)) (use `nvidia-smi` to check your CUDA driver)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```
This should print "True".
- Install this repo
```bash
git clone <repo url>
cd <repo path>
pip install -e .
```
If you want to install the development, visualization and mamba dependencies, you can swap the previous command for the following one:
```bash
pip install -e .[dev,ssm,visu]
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

## Dataset creation
Create a dataset of people, biographies of 1000 peoples, and questions/answers with the following commands (to be run from the root of the repository):
```bash
python -m apps.memory.dataset.generate people
python -m apps.memory.dataset.generate biographies --num 1000
python -m apps.memory.dataset.generate qa --num 100
python -m apps.memory.dataset.generate qa --tooluse --num 100
```
To format the database as a SQLlite database, run the following command
```bash
python -m apps.memory.dataset.sql_db create
```

## Training
Launch a traning run locally
```bash
python -m apps.memory.train apps/memory/config.yaml
```
You can run the code locally with two GPUs (or more).
```bash
torchrun --nproc-per-node 2 -m apps.memory.train apps/memory/config.yaml
```
Launch a training on your cluster
```bash
python -m nanollama.launcher apps/memory/config.yaml
```

#### Notes for the team
I have added two configs `config_with_tool.yaml` and `config_without_tool.yaml` to reproduce Sam's exp1, and continue on our exp1 (get the number of facts a networks can recall with and without access to a tool).
Modify these configs (as well as the datasets -- adding or removing facts) to fit your needs and run the following commands
```bash
python -m apps.memory.train apps/memory/config_with_tool.yaml
python -m apps.memory.train apps/memory/config_without_tool.yaml
```

## Development
Run unit tests with the following command at the root of this repository
```bash
python -m unittest
```

#### Code convention
- Avoid packages that are not well maintained
- If using heavy/hard-to-install packages that are not mandatory, make sure that the code still run if people do not install these packages
- Make sure that the code is open-source-able
- Name `build_<method>` any method that initialize a class.
- Use Object-Oriented Programming, as well as Context-Oriented Programming.
- Make sure that the code can run on CPU and V100 (so that people can easily develop from their own laptop without connection on small datasets).
 - Use Stateful object to be able to relaunch training whenever it crashes.

