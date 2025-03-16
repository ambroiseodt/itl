# Todo:
- Get some curves of accuracy when training with the tool use, or without. and write some visualization notebooks/scripts to get plots out of many experimental logs.

#### Vivien's current todo:

- Relaunch grid as the previous one was faulty due to qa dataset missing answer field.
    - launch the dataset creation (this takes a long time)
    - launch the grid job

- Add an run_config.implementation.{model,profiler} to simplify the parsing of the configs (catch case where `run_cfg.im.profiler: null`), and remove the config dispatcher (in `model.utils`).

- Make nice plots, and visualization methods.
    - Check that wandb is working correctly.
- Generation.
    - Write inference loop logic, as well as sampling strategies.

- Proper parallelization when using many GPUs.
    - TP at training time.
    - DDP at generation time (make KV cache / masking work there).
    - TP at generation time.
    - Check Meta Lingua logic.
    - Check torch.compile(dynamic=True).

- Correct unit tests.
- Check if evals are still working asynchronously.

#### Pipeline modification ideas
- Change mask so that the LLM can have non-causal interaction between tokens that it has not generated.

#### Simple recipes in the apps folder
- Simple vanilla script with real tokenizer over SmolLM v2.
    - Option to remove the tokenizer mask when doing pure pretraining.
- Simple vanilla script on the Shakespear dataset to debug on cpu.

#### Small improvements to the codebase
- (?) Remove Statefulness of scheduler.

#### Known issues
- Issue with compile at inference time.
    - Option to disable compile at generation time (it seems to slow the generation quite a bit).
    - Use `compiled_model = torch.compile(model, dynamic=True)` ?
    - TODO: look at Claude suggestion.
- Parsing configuration starts to be a bit messy. Not clear if there is a clean solution though.
- Wandb restarting does not work well. To have nice eval plots, best is to update local results to wandb after the runs.

#### Bigger improvement to the codebas
- Improve the metric logging, as well as the profiler.
    - log activation norm -> use probing.
        - We may easily log the norm of the weights.
    - have a way to compute exactly the number of flops -> use torch dispatcher.
- Move toward tasks-based evaluation a la llm-harness. Get inspiration from https://github.com/facebookresearch/lingua/blob/main/apps/main/eval.py.
- Improve the generation part to have a async scheme for lane, with page in / page out mechanisms.
    - Caching when dealing with tree of multi-turn dialogs, cpu page-in/page-out mechanism
- Add various initialization scheme.
- Parallelization with (DP, TP) mesh.
    - Understand better the impact of the shard_dim.
    - Check for consolidated checkpointing, as well as saving params.json.
    - Understand interaction with KV cache.
- Activation checkpointing.
- Caching for Mamba and RNN models.
- Make sure one can extend model context size incrementally.

### Ambroise & Sam TODO:
- Read technical reports of open-source LLms (Qwen, DeepSeek, HuggingFace, Olmo, etc.)
- Gain insights on empirical tricks known in the LLM literature to be validated
- Using the codebase, validate (or invalidate) those tricks from the lens of scaling laws
- Important question: given fixed budget (= nb. of params), what is the opitmla design (nb. layers, nb. heads, embd. dim.) to learn the
maximal amount of facts?

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


## Some TODOs of an open-source project:
- torch.compile, int8 quantization, speculative decoding (7x inference acceleration)
- static key-value cache enabled for all HF models (8.5x inference acceleration)
- GPTQ int4 quantization and optimized int4 matmul kernels enabled for all HF models (9x inference acceleration)
- Tensor parallelism + GPU distributed inference
- PagedAttention (vLLM) + FlashAttention integration
- BitNet and 1-bit quantization, AWQ, QoQ, GGUF, HQQ
- Medusa, Speculative Sampling, Eagle
