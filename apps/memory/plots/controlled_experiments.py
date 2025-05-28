"""
Empirical verification of theoretical results with controlled experiments.

@ 2025, Meta
"""

import json
import os
import subprocess
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import rc

from nanollama.utils import flatten_config
from nanollama.visualization.loader import get_config_info, get_task_ids, load_jsonl_to_numpy

logger = getLogger("nanollama")

# ----------------------------------------------------------------------------
# set folder names and plotting params
# ------------------------------------------------------------------------------

# Figure golden ratio (from ICML style file)
WIDTH = 5.5
HEIGHT = 5.5 / 1.5
FONTSIZE = 20

# Tex available
USETEX = not subprocess.run(["which", "pdflatex"], stdout=subprocess.DEVNULL).returncode
USETEX = False

rc("font", family="serif", size=FONTSIZE)
rc("text", usetex=USETEX)
if USETEX:
    rc("text.latex", preamble=r"\usepackage{times}")

RESULT_PATH = Path(__file__).parents[3] / "results"
FIGURE_PATH = Path(__file__).parents[3] / "figures"

# ----------------------------------------------------------------------------
# Utils to save data and load it
# ------------------------------------------------------------------------------


def save_data_from_grid(dirname: str, gridname: str) -> None:
    """Concatenate data coming from a grid and multiple tasks and save it to csv file."""

    # Define log dir
    log_dir = Path.home() / dirname

    # Recover configs
    config_keys = ["data.nb_data", "data.key", "data.seed", "model.emb_dim", "model.nb_layers", "model.block.nb_heads"]
    task_ids = get_task_ids(log_dir)

    # Define steps
    steps = ["best"]

    # Get results
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
    path = RESULT_PATH / gridname
    df.to_csv(path)


def save_data_from_config(exp_name: str) -> None:
    """Concatenate data coming from multiple configs and save it to csv file."""

    # Recover the desired configurations
    task_ids = [1, 5, 6, 7, 8, 9, 14, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    steps = ["best"]

    # Recover configs
    config_keys = ["data.nb_data", "data.key", "data.seed", "model.emb_dim", "model.nb_layers", "model.block.nb_heads"]

    # Get results
    all_res = []
    for task_id in task_ids:
        # Define log dir
        folder_name = f"memory_ood_{task_id}"
        log_dir = Path.home() / folder_name

        res = {}
        log_path = log_dir / "metrics"
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

        # Recover config and model information
        res_config = {}
        config_path = Path(__file__).parents[1] / "config" / f"ood_eval_{str(task_id)}.yaml"
        with open(os.path.expandvars(config_path)) as f:
            config = flatten_config(yaml.safe_load(f))
        for key in config_keys:
            res_config[key] = config[f"run_config.{key}"]

        # number of parameters
        filepath = log_path / "info_model.jsonl"
        with open(os.path.expandvars(filepath)) as f:
            res_config["nb_params"] = json.loads(f.readline())["model_params"]

        # add meta data
        res |= res_config
        all_res.append(res)

        print(f"Task {task_id} done")

    df = pd.DataFrame(all_res)
    path = RESULT_PATH / exp_name
    df.to_csv(path)


def get_data(gridname: str) -> pd.DataFrame:
    """Load data from csv file."""
    path = RESULT_PATH / gridname
    df = pd.read_csv(path)
    return df


def get_in_tool_configs(
    df: pd.DataFrame,
    acc_threshold: float,
) -> None:
    """
    Recover the optimal configurations of the in-tool models, that is the models trained in-tool
    with minimal parameters that at least reach the accuracy threshold.
    """
    x = df["data.nb_data"].unique()

    opt_configs = {}
    all_y_mins = []
    for seed in df["data.seed"].unique():
        opt_configs[seed] = {}
        root_ind = (df["data.seed"] == seed) & (df["accuracy_best"] >= acc_threshold)

        # with tool
        ind = (df["data.key"] == "qatool") & root_ind
        y_min = []
        data = []
        for nb_data in x:
            opt_configs[seed][nb_data] = {}
            restricted_df = df[(df["data.nb_data"] == nb_data) & ind]
            tmp = restricted_df["nb_params"]
            tmp_df = restricted_df[restricted_df["nb_params"] == tmp.min()]
            # print(tmp_df["accuracy_best"])
            data.append(nb_data)
            y_min.append(tmp.min())
            opt_configs[int(seed)][int(nb_data)]["key"] = str(tmp_df["data.key"].values[0])
            opt_configs[int(seed)][int(nb_data)]["emb_dim"] = int(tmp_df["model.emb_dim"].values[0])
            opt_configs[int(seed)][int(nb_data)]["nb_layers"] = int(tmp_df["model.nb_layers"].values[0])
            opt_configs[int(seed)][int(nb_data)]["nb_heads"] = int(tmp_df["model.block.nb_heads"].values[0])
            opt_configs[int(seed)][int(nb_data)]["nb_params"] = int(tmp_df["nb_params"].values[0])
            opt_configs[int(seed)][int(nb_data)]["accuracy_best"] = int(tmp_df["accuracy_best"].values[0])
        all_y_mins.append(y_min)
    data = np.unique(data)
    opt_seeds = np.array(all_y_mins).argmin(axis=0)
    configs = {}
    for seed, nb_data in zip(opt_seeds, data):
        configs[int(nb_data)] = {}
        configs[int(nb_data)]["seed"] = int(seed)
        configs[int(nb_data)]["config"] = opt_configs[int(seed)][int(nb_data)]
    print(configs[2]["config"]["accuracy_best"])
    # Save configs
    config_path = RESULT_PATH / f"configs_in_tool_{int(acc_threshold * 100)}.json"
    with open(config_path, "w") as fp:
        json.dump(configs, fp, ensure_ascii=False, allow_nan=False, indent=4)


# ----------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    if not FIGURE_PATH.exists():
        FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    save_dir = FIGURE_PATH / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def get_combined_legend(ncol: int, loc: str, bbox_to_anchor: tuple, fig: plt.figure, fontsize: int) -> None:
    """Get combined legend for several subplots."""
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fancybox=True,
        borderaxespad=0,
        shadow=True,
        ncol=ncol,
        fontsize=fontsize,
    )


def plot_model_size_nb_facts(
    df: pd.DataFrame,
    acc_threshold: float,
    figname: str,
    tool_label: str = "In-Tool",
    weight_label: str = "In-Weight",
    figsize: tuple = (2 * WIDTH, 0.9 * WIDTH),
    save: bool = True,
    ncol: int = 1,
    loc: str = "upper center",
    bbox_to_anchor: tuple = (0.24, 0.92),
    palette: list = None,
) -> None:
    """
    Plot the evolution of the number of params needed to reach a given level of accuracy
    when the number of facts to learn grows.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    lw = 5
    ms = 10
    alpha = 0.2
    axes = [axes] if not isinstance(axes, np.ndarray) else axes
    x = df["data.nb_data"].unique()

    all_y_mins = []
    all_y_mins_2 = []

    for seed in df["data.seed"].unique():
        root_ind = (df["data.seed"] == seed) & (df["accuracy_best"] >= acc_threshold)

        # with tool
        ind = (df["data.key"] == "qatool") & root_ind
        y_min, y_mean = [], []
        for nb_data in x:
            tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
            y_min.append(tmp.min())
        all_y_mins.append(y_min)

        # without tool
        ind = (df["data.key"] == "qa") & root_ind
        y_min, y_mean = [], []
        for nb_data in x:
            tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
            y_min.append(tmp.min())
        all_y_mins_2.append(y_min)

    y_min = np.array(all_y_mins).min(axis=0)
    y_mean = np.array(all_y_mins).mean(axis=0)
    y_std = np.array(all_y_mins).std(axis=0)
    axes[0].plot(x, y_min, label=tool_label, linestyle="-", linewidth=lw, marker="o", markersize=ms, color=palette[0])
    axes[1].plot(x, y_mean, label=tool_label, linestyle="-", linewidth=lw, marker="o", markersize=ms, color=palette[0])
    axes[1].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=palette[0])

    y_min = np.array(all_y_mins_2).min(axis=0)
    y_mean = np.array(all_y_mins_2).mean(axis=0)
    y_std = np.array(all_y_mins_2).std(axis=0)
    axes[0].plot(x, y_min, label=weight_label, linestyle="-", linewidth=lw, marker="o", markersize=ms, color=palette[1])
    axes[1].plot(
        x, y_mean, label=weight_label, linestyle="-", linewidth=lw, marker="o", markersize=ms, color=palette[1]
    )
    axes[1].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=palette[1])

    # metadata
    axes[0].set_ylabel("Number of Parameters")
    for ax in axes:
        ax.set_xlabel("Number of Facts")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.6, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(direction="out", length=6)
        ax.minorticks_off()

    # Global
    sns.despine(fig, ax, trim=False, right=True, offset=10)

    get_combined_legend(ncol=ncol, loc=loc, bbox_to_anchor=bbox_to_anchor, fig=fig, fontsize=FONTSIZE)

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


def plot_ood_acc_nb_facts(
    df: pd.DataFrame,
    figname: str,
    figsize: tuple = (1.2 * WIDTH, 1.2 * HEIGHT),
    save: bool = True,
    ncol: int = 1,
    loc: str = "upper center",
    palette: list = None,
) -> None:
    """
    Plot the evolution of the OOD accuracy when the number of facts to learn grows for a fixed model's size.
    """

    # Set plotting parameters
    nb_params = [60372, 129752, 591488]
    labels_params = {60372: "60k", 129752: "130k", 591488: "600k"}
    data_threshold = 1024
    labels_method = ["In-Tool", "In-Weight"]

    # Figure parameters
    fig, ax = plt.subplots(figsize=figsize)
    lw = 7

    x = np.sort(df["data.nb_data"].unique())

    for i, key in enumerate(["qatool", "qa"]):
        for j, model_size in enumerate(nb_params):
            # with tool
            ind = (df["data.key"] == key) & (df["nb_params"] == model_size)
            acc = []
            for nb_data in x:
                tmp = df[(df["data.nb_data"] == nb_data) & ind]["accuracy_best"]
                acc.append(tmp.values[0])

            acc = np.asarray(acc)
            ax.plot(
                x,
                acc,
                label=f"{labels_params[model_size]}",
                linestyle="-",
                linewidth=lw,
                color=palette[i, j],
            )

            # metadata
            ax.set_ylabel("OOD Accuracy")
            ax.set_xlabel("Number of Facts")
            ax.set_xscale("log")
            ax.grid(alpha=0.6, lw=1.3)
            ax.spines["left"].set_linewidth(1)
            ax.spines["right"].set_linewidth(1)
            ax.spines["top"].set_linewidth(1)
            ax.spines["bottom"].set_linewidth(1)
            ax.tick_params(direction="out", length=6)
            ax.minorticks_off()

    # Set vline
    ax.axvline(x=data_threshold, linewidth=2.5, linestyle="--", color="black")

    # Global
    sns.despine(fig, ax, trim=False, right=True, offset=10)

    # Set legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    first_legend = fig.legend(
        lines[:3],
        labels[:3],
        loc=loc,
        bbox_to_anchor=(1.15, 0.95),
        fancybox=True,
        borderaxespad=0,
        shadow=True,
        ncol=ncol,
        fontsize=FONTSIZE,
        title=labels_method[0],
        frameon=False,
    )

    fig.legend(
        lines[3:],
        labels[3:],
        loc=loc,
        bbox_to_anchor=(1.15, 0.55),
        fancybox=True,
        borderaxespad=0,
        shadow=True,
        ncol=ncol,
        fontsize=FONTSIZE,
        title=labels_method[1],
        frameon=False,
    )

    ax.add_artist(first_legend)

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


if __name__ == "__main__":
    # Plot evolution of model size with number of facts learned
    # gridname = "grid4.csv"
    # df = get_data(gridname)
    # palette = ["#945785", "#c2a7da"]
    # for acc_threshold in [0.5, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
    #     figname = f"exp0_thresh_{acc_threshold}"
    #     plot_model_size_nb_facts(df=df, acc_threshold=acc_threshold, figname=figname, palette=palette)

    # Get OOD data files
    # dirname = "memory_ood_grid_transition"
    # gridname = "grid_ood_data_transition.csv"
    # save_data_from_grid(dirname=dirname, gridname=gridname)

    # Plot evolution of OOD accuracy with number of facts learned
    gridname = "grid_ood_data_transition.csv"
    df = get_data(gridname)
    palette = np.asarray([["#ccdcef", "#a2b8da", "#5e6d9e"], ["#e3d0ef", "#9d82bf", "#4e4173"]])
    figname = "ood_eval_data_transition"
    plot_ood_acc_nb_facts(df=df, figname=figname, palette=palette)
