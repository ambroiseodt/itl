
Todo:
- Integrate Sam evaluation scripts.
- Get some curves of accuracy when training with the tool use, or without. and write some visualization notebooks/scripts to get plots out of many experimental logs.
- Change a bit the logic for the tool use interaction. Extract the text in ```sql<TEXT>``` and make sure it matches the right text. If so, then we can use the tool use, and we manually inject the database answer.
- ?Use tiktoken tokenizer instead of tokenizing at byte level?

Vivien's todo:
- Understand DeviceMesh. Look into torch (DP, TP) mesh.
    - Distributed Checkpointing.
    - Write the parallelization plan next to the transformer class, try to understand the impact of the shard_dim.
- Generation.
    - KV caching.
    - Prefilling.
    - Parallel completion of prompts with block diagonal mask.
    - Caching when dealing with tree of multi-turn dialogs.
- Parallelization with dp mesh.
    - parallelization of kv cache, parallelization at inference time.
- Tool use mechanism.
    - Regex in token space for database BoS, parsing of previous message, anwser, end of generation.
- Show how to run a grid with a grid.yaml configuration.
- Caching for Mamba and RNN models.
- Add various initialization scheme.
- Improve the light profiler.
- Make sure one can extend model context size incrementally.

Other stuffs for Vivien's
- Simple vanilla script with Byte Tokenizer over Shakespeare.
- Simple vanilla script with real tokenizer over SmolLM v2.
- Implement the tricks that Karpathy present in one of his recent workthrough.
    - Activation checkpointing.

Ambroise's ideas:
- For scaling plot of [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405), could we plot contourlines to take the performance into account?

## Some references
- [ToolFormer](https://arxiv.org/pdf/2302.04761)
- [Memory](https://arxiv.org/pdf/2407.01178v1)
- [Physics of Language Models by Zeyuan Allen-Zhu (tutorial)](https://www.youtube.com/watch?v=yBL7J0kgldU)
- [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405)
- [Zoology: Understand and test LLM on synthetic tasks](https://github.com/HazyResearch/zoology)




alias anode='salloc --account=fair_amaia_cw_codegen --qos=explore --nodes=1 --ntasks-per-node=1 --cpus-per-task=60 --gres=gpu:8 --mem=1700g --time=10:00:00 &'

# Lease a node
salloc --account=fair_amaia_cw_explore --qos=explore --nodes=1 --ntasks-per-node=1 --cpus-per-task=60 --gres=gpu:8 --mem=1700g --time=10:00:00 &

torchrun --nproc_per_node 2 -m apps.rl.train config=apps/rl/configs/r1.yaml

conda activate /checkpoint/amaia/explore/tscohen/envs/amaia-experimental-build-2025-02-17


#### Probing
- Have something running in interactive fashion.
- The logic is that we add module that checkpoint stuff during the forward and during the backward pass, while acting as the identity regarding the information passing through.


Correct init_weight vs reset_parameters
Correct docstring `args` to `- my_args`, and module docstring more readable.