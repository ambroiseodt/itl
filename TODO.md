Todo:
- Modify tiktoken tokenizer to create a special `<tooluse>` token.
- Get better metrics of performance.
- Get some curves, and write some visualization notebooks/scripts to get plots out of many experimental logs.
- Load pretrain models?
- Write a pipeline to first pretrain on `biographies` and then `finetune` on `qa`.

Vivien's todo:
- Write checkpointing logic with the dataloader
- Clean the various monitoring object
- Write a minimal, and a wistle and bell training script.
- Show how to run a grid with a grid.yaml configuration.

Other Viven's todo:
- Caching during generation.
- Probing mechanism.
- Profiling.

Ambroise's ideas:
- For scaling plot of [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405), could we plot contourlines to take the performance into account?

## Notes
Dataloading has some subtleties.
If a sentence is too short compared to the context window, should we concatenate it with another one? should we pad with EOS?
- If we concatenate, should we use some funky masking to avoid paying attention to the previous sentence?
- If we pad, should we reweight the loss to avoid only learning on padding tokens?

## Some references
- [ToolFormer](https://arxiv.org/pdf/2302.04761)
- [Memory](https://arxiv.org/pdf/2407.01178v1)
- [Physics of Language Models by Zeyuan Allen-Zhu (tutorial)](https://www.youtube.com/watch?v=yBL7J0kgldU)
- [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405)
- [Zoology: Understand and test LLM on synthetic tasks](https://github.com/HazyResearch/zoology)