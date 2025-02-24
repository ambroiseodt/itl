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

# from .llama2_model import ModelArgs, Transformer
from nanollama.model import Transformer, TransformerConfig
from nanollama.model.transformer import TransformerBlock, TransformerBlockConfig

# OMP_NUM_THREADS=1 torchrun --nproc-per-node 4 -m src.apps.memory.parallel_debug
# python -m debugpy --wait-for-client --listen 0.0.0.0:5678 `which torchrun` --nproc-per-node 4 -m src.apps.memory.parallel_debug

"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

tp_size = 2
# logger = get_logger()

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")
assert _world_size % tp_size == 0, f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
dp_size = _world_size // tp_size

# Create a device mesh with 2 dimensions.
# First dim is the data parallel dimension
# Second dim is the tensor parallel dimension.
device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
print(f"Device Mesh created: {device_mesh=}")
tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

# For TP, input needs to be same across all TP ranks.
# while for SP, input can be different across all ranks.
# We will use dp_rank for setting the random seed
# to mimic the behavior of the dataloader.
dp_rank = dp_mesh.get_local_rank()

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
simple_llama2_config = TransformerConfig(
    block=TransformerBlockConfig(nb_heads=16, seq_len=256), nb_layers=2, emb_dim=256, vocab_size=32000
)
# simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

model = Transformer(simple_llama2_config).to("cuda")
# model = Transformer.from_model_args(simple_llama2_config).to("cuda")

# init model weights
model.reset_parameters(None, 1)
# model.init_weights()

# parallelize the first embedding and the last linear out projection
SHARD_DIM = 0
model = parallelize_module(
    model,
    tp_mesh,
    {
        "embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(SHARD_DIM),
        ),
        "output_norm": SequenceParallel(sequence_dim=SHARD_DIM),
        "output": ColwiseParallel(input_layouts=Shard(SHARD_DIM), output_layouts=Replicate()),
    },
)

for layer_id, transformer_block in enumerate(model.layers):
    transformer_block: TransformerBlock
    layer_tp_plan = {
        "attn_norm": SequenceParallel(sequence_dim=SHARD_DIM),
        "attn": PrepareModuleInput(
            input_layouts=(Shard(SHARD_DIM), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attn.W_query": ColwiseParallel(input_layouts=Shard(SHARD_DIM)),
        "attn.W_key": ColwiseParallel(input_layouts=Shard(SHARD_DIM)),
        "attn.W_val": ColwiseParallel(input_layouts=Shard(SHARD_DIM)),
        "attn.W_out": RowwiseParallel(output_layouts=Shard(SHARD_DIM)),
        "ffn_norm": SequenceParallel(sequence_dim=SHARD_DIM),
        "ffn": PrepareModuleInput(
            input_layouts=(Shard(SHARD_DIM),),
            desired_input_layouts=(Replicate(),),
        ),
        "ffn.W_in1": ColwiseParallel(input_layouts=Shard(SHARD_DIM)),
        "ffn.W_in2": ColwiseParallel(input_layouts=Shard(SHARD_DIM)),
        "ffn.W_out": RowwiseParallel(output_layouts=Shard(SHARD_DIM)),
    }
    # layer_tp_plan = {
    #     "attention_norm": SequenceParallel(),
    #     "attention": PrepareModuleInput(
    #         input_layouts=(Shard(1), None),
    #         desired_input_layouts=(Replicate(), None),
    #     ),
    #     "attention.wq": ColwiseParallel(),
    #     "attention.wk": ColwiseParallel(),
    #     "attention.wv": ColwiseParallel(),
    #     "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    #     "ffn_norm": SequenceParallel(),
    #     "feed_forward": PrepareModuleInput(
    #         input_layouts=(Shard(1),),
    #         desired_input_layouts=(Replicate(),),
    #     ),
    #     "feed_forward.w1": ColwiseParallel(),
    #     "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    #     "feed_forward.w3": ColwiseParallel(),
    # }

    # Adjust attention module to use the local number of heads
    attn_layer = transformer_block.attn
    attn_layer.nb_heads = attn_layer.nb_heads // tp_mesh.size()
    attn_layer.nb_kv_heads = attn_layer.nb_kv_heads // tp_mesh.size()
    # attn_layer = transformer_block.attention
    # attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
    # attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

    # Custom parallelization plan for the model
    parallelize_module(module=transformer_block, device_mesh=tp_mesh, parallelize_plan=layer_tp_plan)

# Init FSDP using the dp device mesh
sharded_model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)

# rank_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")
print(f"Model after parallelization {sharded_model=}\n")

# Create an optimizer for the parallelized and sharded model.
lr = 3e-3
# rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
print(f"Creating AdamW optimizer with learning rate {lr}")
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)

# Training loop:
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
# rank_log(_rank, logger, "\nStarting 2D training...")
print("\nStarting 2D training...")
num_iterations = 10
batch_size = 2

for i in range(num_iterations):
    # seeding with dp_rank to ensure identical inputs for TP groups
    torch.manual_seed(i + dp_rank)
    inp = torch.randint(32000, (8, 256), device="cuda")

    output = sharded_model(inp)
    output.sum().backward()
    optimizer.step()
    # rank_log(_rank, logger, f"2D iter {i} complete")
    print(f"2D iter {i} complete")

# rank_log(_rank, logger, "2D training successfully completed!")
print("2D training successfully completed!")
