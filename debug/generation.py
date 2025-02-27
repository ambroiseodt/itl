# %%
import torch
import torch.distributed.checkpoint as dcp
import yaml

from nanollama.data.loader import DataLoader
from nanollama.data.text import DataConfig, MultipleSourcesTokenGenerator
from nanollama.model import Transformer, TransformerConfig
from nanollama.utils import initialize_nested_object

# get some data
config = yaml.safe_load("""
sources:
- path: $HOME/code/memory/src/apps/memory/dataset/qa.jsonl
  weight: 10
- path: $HOME/code/memory/src/apps/memory/dataset/qatool.jsonl
  weight: 50
tokenizer:
    name: byte
padding: true
batch_size: 16
seq_len: 257
asynchronous: false
""")
data_config = initialize_nested_object(DataConfig, config)

token_gen = MultipleSourcesTokenGenerator(data_config)
dataloader = DataLoader(data_config, token_gen)
with dataloader:
    batch = next(dataloader)

data, mask = batch.chunk(2)
prefix_lens = mask.argmax(dim=1)

print("printing data")
decode_func = token_gen.generators[0].tokenizer.decode
for x, prefix in zip(data, prefix_lens):
    print(decode_func(x))
    print(decode_func(x[:prefix]))

# get a model
path = "/private/home/vivc/memory/checkpoints/0/0000001000"
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

# %%

model.build_cache(bsz=data.size(0))
x, doc_start = model.build_prompts(data, prefix_lens)

print("\n\nprinting prompt fed to the model")
for s in x:
    print(decode_func(s))

seq_len = x.size(1)
preds, seq = [], [x]
pred_len = 0

with torch.inference_mode():
    while seq_len <= 256:
        pred = model(x, doc_start=doc_start)
        x = pred[:, -1:].argmax(dim=2)
        preds.append(pred)
        seq.append(x)
        seq_len += 1
        doc_start = None

seq = torch.hstack(seq).tolist()
for s in seq:
    print(decode_func(s))
preds = torch.hstack(preds)

# %%

model.build_cache(bsz=data.size(0))
new_data = torch.tensor(seq, dtype=torch.long)[:, :-1]

with torch.inference_mode():
    new_preds = model(new_data)

print((preds - new_preds).max())
print(torch.allclose(preds, new_preds, atol=1e-5))

# %%
