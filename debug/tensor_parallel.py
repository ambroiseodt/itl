import os

import torch
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from nanollama.model.transformer import (  # Replace 'your_module' with the actual module name
    Transformer,
    TransformerBlockConfig,
    TransformerConfig,
)

# Initialize distributed environment
tp_size = 1
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])

assert _world_size % tp_size == 0, f"World size {_world_size} needs to be divisible by TP size {tp_size}"

dp_size = _world_size // tp_size
device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

# Configuration for the transformer model
transformer_config = TransformerConfig(
    emb_dim=256,
    vocab_size=32_000,
    nb_layers=2,
    block=TransformerBlockConfig(seq_len=256, nb_heads=16),
)

# Initialize the model
model = Transformer(transformer_config).to("cuda")

# Example of a parallelization plan for a module with a single input
tp_plan = {
    "embeddings": RowwiseParallel(
        input_layouts=Replicate(),  # Ensure this matches the number of inputs
        output_layouts=Shard(1),
    ),
    "output_norm": SequenceParallel(),
    "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
}
for layer_id in range(len(model.layers)):
    tp_plan |= {
        f"layers.{layer_id}.attn_norm": SequenceParallel(),
        f"layers.{layer_id}.attn": PrepareModuleInput(
            input_layouts=(Shard(1)),  # Ensure this matches the number of inputs
            desired_input_layouts=(Replicate()),
        ),
        f"layers.{layer_id}.attn.W_query": ColwiseParallel(),
        f"layers.{layer_id}.attn.W_key": ColwiseParallel(),
        f"layers.{layer_id}.attn.W_val": ColwiseParallel(),
        f"layers.{layer_id}.attn.W_out": RowwiseParallel(output_layouts=Shard(1)),
        f"layers.{layer_id}.ffn_norm": SequenceParallel(),
        f"layers.{layer_id}.ffn": PrepareModuleInput(
            input_layouts=(Shard(1),),  # Ensure this matches the number of inputs
            desired_input_layouts=(Replicate(),),
        ),
        f"layers.{layer_id}.ffn.W_in1": ColwiseParallel(),
        f"layers.{layer_id}.ffn.W_in2": ColwiseParallel(),
        f"layers.{layer_id}.ffn.W_out": RowwiseParallel(output_layouts=Shard(1)),
    }

parallelize_module(model, tp_mesh, tp_plan)

# Initialize FSDP
sharded_model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
print(f"Model after parallelization {sharded_model=}\n")

# Create an optimizer
lr = 3e-3
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)

# Training loop
num_iterations = 10
batch_size = 2

for i in range(num_iterations):
    torch.manual_seed(i + dp_mesh.get_local_rank())
    inp = torch.randint(32000, (8, transformer_config.block.seq_len), device="cuda")
    output = sharded_model(inp)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    print(f"2D iter {i} complete {loss.item()}")

print("2D training successfully completed!")
