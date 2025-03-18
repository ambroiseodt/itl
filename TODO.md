#### Vivien's current todo:

- Get a nice Figure 1.
    - I have relaunched a finer grid to get a nicer Figure 1.
    - What is the best allocation of parameters given a budget to maximize performance (more layers, more emb dim, how many heads..., two layers seems best for tool use, otherwise things are unclear), maybe we can answer some of these question theoretically.
    - Does the number of bits of storage we can store depends on the quantization of the parameters (the best we could hope for would be a one to one mapping between nb bytes in network parameters and minimal nb_bytes to compress the data). This will be hard to answer with accessing V100 only.
    - **Mixture of experts** would be quite useful to basically scale in terms of facts memorized according to the number of total parameters, while only paying the inference cost for the number of active parameters. This sounds quite promising paper-wise.

- Do some analysis of the circuit evolution for the runs 1161.

#### Simple recipes in the apps folder
- Simple vanilla script with real tokenizer over SmolLM v2.
    - Option to remove the tokenizer mask when doing pure pretraining.
        - Do this by extracting the json loader, then splitting between DialogTokenGenerator and TokenGenerator (without dialog). Put the three files in data/text/...
- Simple vanilla script on the Shakespear dataset to debug on cpu.

#### Known issues
- Wandb restarting does not work well. To have nice eval plots, best is to update local results to wandb after the runs.
- Restart after an interruption during eval when launched online are not handled correctly.

#### Bigger improvement to the codebase
**Stuffs to improve and understand pretraining:**
- Add various initialization scheme.
- Improve the metric logging.
    - log activation norm -> use probing.
        - We may easily log the norm of the weights.
    - have a way to compute exactly the number of flops -> use torch dispatcher.

**Stuffs to benchmark real model:**
- Move toward tasks-based evaluation a la llm-harness. Get inspiration from https://github.com/facebookresearch/lingua/blob/main/apps/main/eval.py.
- Caching for Mamba and RNN models.
- Write various sampling strategies.

**Stuffs for fast generation in RL like env:**
- Improve the generation part to have a async scheme for lane, with page in / page out mechanisms.
    - Indeed, generation is currently bottlenecked by the SQL agent.
    - Caching when dealing with tree of multi-turn dialogs, cpu page-in/page-out mechanism.

**Stuffs to scale models:**
- Tinker with parallelization when using many GPUs.
    - TP at training time.
    - DDP at generation time (make KV cache / masking work there).
    - TP at generation time.
    - Check Meta Lingua logic.
    - Check torch.compile(dynamic=True).
- Activation checkpointing.

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

#### Pipeline modification ideas
- Change mask so that the LLM can have non-causal interaction between tokens that it has not generated.

## Some TODOs of an open-source project:
- torch.compile, int8 quantization, speculative decoding (7x inference acceleration)
- static key-value cache enabled for all HF models (8.5x inference acceleration)
- GPTQ int4 quantization and optimized int4 matmul kernels enabled for all HF models (9x inference acceleration)
- Tensor parallelism + GPU distributed inference
- PagedAttention (vLLM) + FlashAttention integration
- BitNet and 1-bit quantization, AWQ, QoQ, GGUF, HQQ
- Medusa, Speculative Sampling, Eagle


## Potential story
1. In-tool allows you to reduce the memory footprint of the model, you learn rules not facts.
    - Figure 1: we show some nice scaling for nb facts learned vs nb of parameters (without tool use this is linear, with tool use we have a much better scaling, basically limited by copying ability, maybe log scaling - we could do theory there).
2. To learn rules, you need to bypass the tendancy of the model to memorize facts, which helps descrease the learning rate faster.
    - Show the circuits, show the learning dynamics (if we slow down the learning rate, can we avoid the memorization phase, can this be achieved with the warm-up schedule?)
