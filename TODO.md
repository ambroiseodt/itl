# Todo:
- Integrate Sam evaluation scripts.
- Get some curves of accuracy when training with the tool use, or without. and write some visualization notebooks/scripts to get plots out of many experimental logs.

#### Vivien's current todo:
- Code the `batch_queue` class.
- Make a `agent_eval.py` file in the memory apps to evaluate text generate with tool-use.
- Generation.
    - Write inference loop logic, as well as sampling strategies.

#### Other stuffs for Vivien's
- Improve the light profiler.
- Improve logging in the train loop, and add visualization of training logs.
- Linear Probing of the weights of the network
    - Have something running in interactive fashion.
    - The logic is that we add module that checkpoint stuff during the forward and during the backward pass, while acting as the identity regarding the information passing through.

#### Pipeline modification ideas
- Change mask so that the LLM can have non-causal interaction between tokens that it has not generated.

#### Simple recipes in the apps folder
- Simple vanilla script with Byte Tokenizer over Shakespeare.
- Simple vanilla script with real tokenizer over SmolLM v2.
- Show how to run a grid with a grid.yaml configuration.

#### Cosmetic changes
- Correct init_weight vs reset_parameters
- Correct docstring `args` to `- my_args`, and module docstring more readable.
- Improve the docstring at module level (the ones at the start of files).

#### Small improvements to the codebase
- (?) Remove Statefulness of scheduler.
- Option to remove the tokenizer mask when doing pure pretraining.

#### Bigger improvement to the codebase
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
