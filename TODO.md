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

## Notes
Dataloading has some subtleties.
If a sentence is too short compared to the context window, should we concatenate it with another one? should we pad with EOS?
- If we concatenate, should we use some funky masking to avoid paying attention to the previous sentence?
- If we pad, should we reweight the loss to avoid only learning on padding tokens?
