Todo:
- Integrate Sam evaluation scripts.
- Get some curves of accuracy when training with the tool use, or without. and write some visualization notebooks/scripts to get plots out of many experimental logs.
- Change a bit the logic for the tool use interaction. Extract the text in ```sql<TEXT>``` and make sure it matches the right text. If so, then we can use the tool use, and we manually inject the database answer.
- ?Use tiktoken tokenizer instead of tokenizing at byte level?

Vivien's todo:
- Write generation part: caching mechanism, tool use mechanism.
- Improve the light profiler.
- Show how to run a grid with a grid.yaml configuration.

Ambroise's ideas:
- For scaling plot of [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405), could we plot contourlines to take the performance into account?

## Some references
- [ToolFormer](https://arxiv.org/pdf/2302.04761)
- [Memory](https://arxiv.org/pdf/2407.01178v1)
- [Physics of Language Models by Zeyuan Allen-Zhu (tutorial)](https://www.youtube.com/watch?v=yBL7J0kgldU)
- [Physics of Language Models: Part 3.3](https://arxiv.org/pdf/2404.05405)
- [Zoology: Understand and test LLM on synthetic tasks](https://github.com/HazyResearch/zoology)