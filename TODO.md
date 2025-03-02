
Todo:
- Integrate Sam evaluation scripts.
- Get some curves of accuracy when training with the tool use, or without. and write some visualization notebooks/scripts to get plots out of many experimental logs.
- Change a bit the logic for the tool use interaction. Extract the text in ```sql<TEXT>``` and make sure it matches the right text. If so, then we can use the tool use, and we manually inject the database answer.

Vivien's todo:
- Agentic interaction:
    - write a simple script that do the interaction with a mock SQL agent.
    - implement a SQL agent module.
    - make a class that deal with the generation at a more abstract level / or an agentic app (in the `apps` folder).
        - tooluse mechanism: regex in token space for database BoS, parsing of previous message, anwser, end of generation.
- Generation.
    - Write inference loop logic, as well as sampling strategies.
- Show how to run a grid with a grid.yaml configuration.

Other stuffs for Vivien's
- Improve the light profiler.
- Simple vanilla script with Byte Tokenizer over Shakespeare.
- Simple vanilla script with real tokenizer over SmolLM v2.
- Correct init_weight vs reset_parameters
- Correct docstring `args` to `- my_args`, and module docstring more readable.
- Remove Statefulness of scheduler.
- Option to remove the tokenizer mask when doing pure pretraining.
- Change mask so that the LLM can have non-causal interaction between tokens that it has not generated.
- Improve logging in the train loop, and add visualization of training logs.

Ambroise's ideas:
- For scaling plot of [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405), could we plot contourlines to take the performance into account?

## Some references
- [ToolFormer](https://arxiv.org/pdf/2302.04761)
- [Memory](https://arxiv.org/pdf/2407.01178v1)
- [Physics of Language Models by Zeyuan Allen-Zhu (tutorial)](https://www.youtube.com/watch?v=yBL7J0kgldU)
- [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405)
- [Zoology: Understand and test LLM on synthetic tasks](https://github.com/HazyResearch/zoology)

#### Probing
- Have something running in interactive fashion.
- The logic is that we add module that checkpoint stuff during the forward and during the backward pass, while acting as the identity regarding the information passing through.

#### Potential points to better understand
- Add various initialization scheme.

- Parallelization with (DP, TP) mesh.
    - Understand better the impact of the shard_dim.
    - Check for consolidated checkpointing, as well as saving params.json.
    - Understand interaction with KV cache.

- Activation checkpointing.

- Caching when dealing with tree of multi-turn dialogs, cpu page-in/page-out mechanism
- Prefix attention masking 

- Caching for Mamba and RNN models.

- Make sure one can extend model context size incrementally.

- Implement logic for tiktoken, and sentencepiece.
