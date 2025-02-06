Fill the different parts of the code, with inspiration from nanoGPT, Meta Lingua, and Meta PAL.
Have minimal working pipelines.

Minimal working pipelines.

- Test the asynchronous dataloader.
- write a training loop.


- Structure for the memory project.
	- Dataloader and tokenizer.
		- Complete the dataloading
		- Put the Tiktoken tokenizer
	- Put back all the monitoring logic from nanollama.
	- Pretrained model (?).
	
- Caching during generation.

- Finetuning logic.

- Probing mechanism.

## Notes
Dataloading has some subtleties.
If a sentence is too short compared to the context window, should we concatenate it with another one? should we pad with EOS?
- If we concatenate, should we use some funky masking to avoid paying attention to the previous sentence?
- If we pad, should we reweight the loss to avoid only learning on padding tokens?
