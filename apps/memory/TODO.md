### TO DO

## Controlled Exp
1) Improve first plot version of Vivien (nicer with less plateaus)
2) Recover the optimal in-tools models (on the plateau), rerun those configs to keep the checkpoints and evaluate them on an OOD dataset (e.g. people.db made out of names not present in the training data).
    - We would like it to be good in OOD, showing that the plateaus means that in-tool models are not learning fact anymore but are in copy/generalization mode.
    - Maybe do the same with optimal in-tool configs but trained in in-weight and we hope it does not generalize well

## Large-scale exp
1) Finetuning pipeline with hellaswag stopping condition using HF Trainer
2) Same as exp0 with different sizes of GPT2, Llama 3 (1B/3B/7B)
3) Explore impact of QLoRa --> enables to make the size vary more
    - Take Llama 1B and run 10 finetuning with quantization 1%, 2%, 5%, etc.
    - Plot the performance loss on hellaswag as a function of nb of training data for those different sizes