import torch.distributed.checkpoint as dcp
import yaml

from nanollama.data.loader import DataLoader
from nanollama.data.text import DataConfig, MultipleSourcesTokenGenerator
from nanollama.inference import QueuedBatchedInference
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
decode_func = token_gen.generators[0].tokenizer.tokenizer.decode
for datum, dlen in zip(data, prefix_lens):
    prompts.append(decode_func(datum[:dlen]))
    print(prompts[-1], "\n")

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


DB_PATH = "/private/home/vivc/code/memory/apps/memory/dataset/people.db"
model = model.to("cuda")
tokenizer = token_gen.generators[0].tokenizer
inference_engine = QueuedBatchedInference(model, tokenizer, DB_PATH)

with inference_engine:
    outputs = inference_engine.generate(prompts)

for p, o in zip(prompts, outputs):
    print(p, "\n", o, "\n\n")
