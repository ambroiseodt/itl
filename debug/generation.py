from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import yaml

from nanollama.agents import SQLAgent
from nanollama.data.loader import DataLoader
from nanollama.data.text import DataConfig, MultipleSourcesTokenGenerator
from nanollama.model import Transformer, TransformerConfig
from nanollama.model import transformer as tf
from nanollama.utils import initialize_nested_object

tf.FLEX_ATTENTION = False

# get some data
config = yaml.safe_load("""
sources:
- path: $HOME/code/memory/apps/memory/dataset/qatool.jsonl
  weight: 50
tokenizer:
    name: byte
padding: true
batch_size: 8
seq_len: 257
asynchronous: false
""")
data_config = initialize_nested_object(DataConfig, config)

token_gen = MultipleSourcesTokenGenerator(data_config)
dataloader = DataLoader(data_config, token_gen)
with dataloader:
    batch = next(dataloader)

data, mask = batch.chunk(2)
prompts = []
prefix_lens = mask.argmax(dim=1) + 1

print("printing data")
decode_func = token_gen.generators[0].tokenizer.decode
for datum, dlen in zip(data, prefix_lens):
    prompts.append(datum[:dlen])
    print(decode_func(prompts[-1]), "\n")

# get a model
path = "/private/home/vivc/memory/checkpoints/0/0000002000"
config = yaml.safe_load("""
vocab_size: 300
emb_dim: 64
nb_layers: 2
block:
    seq_len: 256
    nb_heads: 2
    hidden_dim: 256
""")

model = Transformer(initialize_nested_object(TransformerConfig, config))

state_dict = {"model": model.state_dict()}
dcp.load(state_dict=state_dict, checkpoint_id=path)
model.load_state_dict(state_dict["model"])
model = model.to("cuda")
prompts = [p.to("cuda") for p in prompts]
tokenizer = token_gen.generators[0].tokenizer


SAVE_DIR = Path.home() / "code" / "memory" / "apps" / "memory" / "dataset"
with SQLAgent(SAVE_DIR / "people.db") as sql_agent:
    for i in range(len(prompts)):
        # prefilling
        _prompts = prompts[i : i + 1]
        x = model.setup_inference(_prompts)
        pred = model(x)

        seq_len = x.size(1)
        nb_prompts = len(_prompts)
        buffers = [[] for _ in range(nb_prompts)]
        with torch.inference_mode():
            while seq_len < 256:
                token = pred[:, -1:].argmax(dim=2)
                seq_len += 1

                for i in range(nb_prompts):
                    actor = tokenizer.bot2actor.get(token[i].item(), None)
                    if actor is not None:
                        output = tokenizer.tokenizer.decode(buffers[i])
                        buffers[i] = []
                        print(f"Calling {actor} with prompt: {output}")
                        if actor == sql_agent.actor:
                            answer = sql_agent.execute(output)
                            print(f"the answer is {answer}")
                    else:
                        buffers[i].append(token.item())

                pred = model(token)

        # seq = torch.tensor(buffers).tolist()
        # for p, s in zip(_prompts, seq):
        #     print("NEW ANSWER")
            # print(decode_func(p))
            # print(decode_func(s))
