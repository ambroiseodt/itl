# %%
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nanollama.visualization.loader import get_config_info, get_task_ids, load_jsonl_to_numpy

logger = getLogger("nanollama")


# %%----------------------------------------------------------------------------
# Assuming you have run a grid, you can use the following code to concatenate the results
# ------------------------------------------------------------------------------

# put the path to the logging directory of your runs
log_dir = Path.home() / "memory_grid"

# steps = [20, 100, 1000, 10000]
steps = ["best"]

task_id = 1
config_keys = ["data.nb_data", "data.key", "model.emb_dim", "model.nb_layers", "model.block.nb_heads"]

task_ids = get_task_ids(log_dir)
all_res = []
for task_id in task_ids:
    res = {}
    log_path = log_dir / "metrics" / str(task_id)
    data_keys = ["loss", "step", "accuracy"]
    data = load_jsonl_to_numpy(log_path / "raw_0.jsonl", keys=data_keys)

    step = data["step"]
    for snapshot in steps:
        try:
            if snapshot == "best":
                res[f"loss_{snapshot}"] = np.nanmin(data["loss"].astype(float))
                res[f"accuracy_{snapshot}"] = np.nanmax(data["accuracy"].astype(float))
            else:
                idx = (step == snapshot).argmax()
                res[f"loss_{snapshot}"] = data["loss"][idx]
                res[f"accuracy_{snapshot}"] = data["accuracy"][idx + 1]
        except Exception as e:
            logger.error(f"Error extracting loss for snapshot {snapshot}: {str(e)}")

    # add meta data
    res |= get_config_info(log_dir, task_id, keys=config_keys, num_params=True)
    all_res.append(res)

    print(f"Task {task_id} done")

df = pd.DataFrame(all_res)


# %%----------------------------------------------------------------------------
# Otherwise you can load some results that I have saved in the repo
# ------------------------------------------------------------------------------

path = Path(__file__).parents[3] / "results" / "grid3.csv"
# df.to_csv(path)
df = pd.read_csv(path)


# %%----------------------------------------------------------------------------
# From the dataframe, you can plot various things, I go for some basic plots here
# ------------------------------------------------------------------------------

xaxis = "data.nb_data"
yaxis = "accuracy_best"
keys = ["model.emb_dim", "model.nb_layers", "model.block.nb_heads"]
for key in ["qa", "qatool"]:
    ind = df["data.key"] == key
    filtered_df = df[ind][["data.key", xaxis, yaxis] + keys]

    grouped = filtered_df.groupby(keys)
    # Plotting
    plt.figure(figsize=(10, 6))
    for name, group in grouped:
        plt.plot(group[xaxis], group[yaxis], marker="o", linestyle="-", label=f"{name}")
    plt.title(f"Performance for {key}")
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.xscale("log")
    # plt.legend(title="(data.key, emb_dim, nb_layers, nb_heads)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%----------------------------------------------------------------------------
# Add a measure of facts memorized
# ------------------------------------------------------------------------------

df["nb_facts"] = df["data.nb_data"] * df["accuracy_best"]


# %%----------------------------------------------------------------------------
# Show the number of learned facts vs the number of parameters
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

nb_data = df["data.nb_data"].max()

# with tool
ind = (df["data.key"] == "qatool") & (df["data.nb_data"] == nb_data)
tmp = df[ind]
ax.scatter(tmp["nb_facts"], tmp["nb_params"], label="With tool", alpha=0.7)

# without tool
ind = (df["data.key"] == "qa") & (df["data.nb_data"] == nb_data)
tmp = df[ind]
ax.scatter(tmp["nb_facts"], tmp["nb_params"], label="Without tools", alpha=0.7)

# metadata
ax.set_title("Number of learned facts (on 10,000)")
ax.set_xlabel("Number of facts")
ax.set_ylabel("Number of parameters")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

# Show plot
plt.tight_layout()
plt.show()


# %%----------------------------------------------------------------------------
# Similar plot that is crisper
# ------------------------------------------------------------------------------

# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig, axes = plt.subplots(1, 1, figsize=(4, 4))
axes = [axes] if not isinstance(axes, list) else axes
x = df['data.nb_data'].unique()

acc_thres = 0.9

# with tool
ind = (df["data.key"] == "qatool") & (df["accuracy_best"] >= acc_thres)
y_min, y_mean = [], []
for nb_data in x:
    tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
    y_min.append(tmp.min())
    # y_mean.append(tmp.mean())
axes[0].plot(x, y_min, label="With tool", linestyle="--", marker="o")
# axes[1].plot(x, y_mean, label="With tool", linestyle="--", marker="o")

# without tool
ind = (df["data.key"] == "qa") & (df["accuracy_best"] >= acc_thres)
y_min, y_mean = [], []
for nb_data in x:
    tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
    y_min.append(tmp.min())
    # y_mean.append(tmp.mean())
axes[0].plot(x, y_min, label="Without tool", linestyle="--", marker="o")
# axes[1].plot(x, y_mean, label="Without tool", linestyle="--", marker="o")

# metadata
axes[0].set_ylabel(f"Min nb params s.t. accuracy > {acc_thres}")
# axes[1].set_ylabel(f"Mean nb params s.t. accuracy > {acc_thres}")
for ax in axes:
    ax.set_xlabel("Number of data")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

# Show plot
plt.tight_layout()
plt.show()


# %%----------------------------------------------------------------------------
# Look at the influence of the number of heads
# ------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
x = df['data.nb_data'].unique()

acc_thres = 0.9

# key = "model.block.nb_heads"
# key = "model.nb_layers"
key = "model.emb_dim"
val = df[key].unique()

# with tool
for i in val:
    params_ind = df[key] == i
    ind = (df["data.key"] == "qatool") & (df["accuracy_best"] >= acc_thres)
    ind &= params_ind
    y_min, y_mean = [], []
    for nb_data in x:
        tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
        y_min.append(tmp.min())
    axes[0].plot(x, y_min, label=f"With tool, {i} {key}", linestyle="--", marker="o")

    # without tool
    ind = (df["data.key"] == "qa") & (df["accuracy_best"] >= acc_thres)
    ind &= params_ind
    y_min, y_mean = [], []
    for nb_data in x:
        tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
        y_min.append(tmp.min())
    axes[1].plot(x, y_min, label=f"Without tool, {i} {key}", linestyle="--", marker="o")

# metadata
axes[0].set_ylabel(f"Min nb params s.t. accuracy > {acc_thres}")
axes[1].set_ylabel(f"Mean nb params s.t. accuracy > {acc_thres}")
for ax in axes:
    ax.set_xlabel("Number of data")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

# Show plot
plt.tight_layout()
plt.show()

# %%
