# In-Tool Learning
This codebase aims to study the "In-Tool Learning" of Large Language Models.

- In-Tool Learning: Learning to use a tool (e.g., a calculator or a request to a database) to answer the problem,
- In-Weight Learning: Memorizing the solution to the prolem within the model's weights.

## Installation
Please, make sure you have Python 3.10 or a newer version installed.

It is preferred that the library be installed in a virtual environment, for instance with conda and Python 3.9:
```bash
conda create -n myenv python==3.10
conda activate myenv
```

If you only want use the library, you can install the latest version of the code with:
```bash
pip install git+https://github.com/viviencabannes/memory.git#egg=draft
```

If you want to contribute, you can clone the repository and install the packages as follows:
```bash
git clone https://github.com/viviencabannes/memory.git
cd memory
pip install -e .
```

## Dataset creation
Create a dataset of people, biographies of 1000 peoples, and questions/answers with the following commands (to be run from the root of the repository):
```bash
python -m src.apps.memory.dataset.generate people
python -m src.apps.memory.dataset.generate biographies --num 1000
python -m src.apps.memory.dataset.generate qa --tooluse
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

