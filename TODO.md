# Todo:
- Integrate Sam evaluation scripts.
- Get some curves of accuracy when training with the tool use, or without. and write some visualization notebooks/scripts to get plots out of many experimental logs.

#### Vivien's current todo:

- Correct the unit tests

- Show how to run a grid with a grid.yaml configuration.

- Launch some big grids to get first plots.

- improve the metric logging, as well as the profiler.
    - log learning rate.
    - log gradient norm.
    - log activation norm.
    - log gpu utilization.
    - log throughput.
    - have a way to compute exactly the number of flops.

- Make nice plots, and visualization methods.
    - check that wandb is working correctly.
- Generation.
    - Write inference loop logic, as well as sampling strategies.

- Proper parallelization when using many GPUs.
    - DDP at generation time (make KV cache / masking work there).
    - TP at generation time.

#### Other stuffs for Vivien's
- Improve the light profiler.

- Linear Probing of the weights of the network
    - Have something running in interactive fashion.
    - The logic is that we add module that checkpoint stuff during the forward and during the backward pass, while acting as the identity regarding the information passing through.

#### Pipeline modification ideas
- Change mask so that the LLM can have non-causal interaction between tokens that it has not generated.

#### Simple recipes in the apps folder
- Simple vanilla script with real tokenizer over SmolLM v2.
    - Option to remove the tokenizer mask when doing pure pretraining.

#### Small improvements to the codebase
- (?) Remove Statefulness of scheduler.
- It is a bit weird to mix the metric logger with the stdout logger.

#### Bigger improvement to the codebas
- Option to disable compile at generation time (it seems to slow the generation quite a bit).
- Move toward omegaconf to handle configurations with `__post_init__` becoming `check_init`.
- Move toward tasks-based evaluation a la llm-harness. Get inspiration from https://github.com/facebookresearch/lingua/blob/main/apps/main/eval.py.
- Improve the generation part to have a async scheme for lane, with page in / page out mechanisms.
- Add various initialization scheme.
- Parallelization with (DP, TP) mesh.
    - Understand better the impact of the shard_dim.
    - Check for consolidated checkpointing, as well as saving params.json.
    - Understand interaction with KV cache.
- Activation checkpointing.
- Caching when dealing with tree of multi-turn dialogs, cpu page-in/page-out mechanism
- Caching for Mamba and RNN models.
- Make sure one can extend model context size incrementally.
- Implement logic for tiktoken, and sentencepiece.

# Ambroise's ideas:
- For scaling plot of [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405), could we plot contourlines to take the performance into account?
- Adapt Sam evaluation files

## Some references
- [ToolFormer](https://arxiv.org/pdf/2302.04761)
- [Memory](https://arxiv.org/pdf/2407.01178v1)
- [Physics of Language Models by Zeyuan Allen-Zhu (tutorial)](https://www.youtube.com/watch?v=yBL7J0kgldU)
- [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405)
- [Zoology: Understand and test LLM on synthetic tasks](https://github.com/HazyResearch/zoology)

## Generic concepts
- Mixture of experts.
- Low rank finetuning.
- Quantization.
