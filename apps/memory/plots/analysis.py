"""
Empirical verification of theoretical results with controlled experiments.

@ 2025, Meta
"""

import json
import logging
import os
import subprocess
from logging import getLogger
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import rcParams

from nanollama.utils import flatten_config
from nanollama.visualization.loader import get_config_info, get_task_ids, load_jsonl_to_numpy

logger = getLogger("nanollama")

# ----------------------------------------------------------------------------
# set folder names and plotting params
# ------------------------------------------------------------------------------

# Figure golden ratio (from ICML style file)
WIDTH = 3.5
HEIGHT = WIDTH / 1.5
FONTSIZE = 10
MARKER_SIZE = 6
LINEWIDTH = 3.5
ALPHA_GRID = 0.2

# Plotting format
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.75,
        "lines.markersize": 4,
        "axes.edgecolor": "black",
        "axes.facecolor": "#f7f7f7",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "gray",
        "legend.framealpha": 1.0,
        "savefig.dpi": 300,
    })

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


def save_compressed_data_config(gridname: str) -> None:
    """Concatenate data coming from multiple configs and save it to csv file."""

    # Recover the desired configurations
    task_ids = range(55)
    steps = ["best"]

    # Recover configs
    config_keys = ["data.nb_data", "data.key", "data.seed", "model.emb_dim", "model.nb_layers", "model.block.nb_heads"]

    # Get results
    all_res = []
    for task_id in task_ids:
        # Define log dir
        folder_name = f"memory_dependent_run_{task_id}"
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
        config_path = Path(__file__).parents[1] / "config" / "compressibility" / f"run_{str(task_id)}.yaml"
        with open(os.path.expandvars(config_path)) as f:
            config = flatten_config(yaml.safe_load(f))
        for key in config_keys:
            res_config[key] = config[f"run_config.{key}"]

        # number of parameters
        filepath = log_path / "info_model.jsonl"
        with open(os.path.expandvars(filepath)) as f:
            res_config["nb_params"] = json.loads(f.readline())["model_params"]

        # Alpha
        string = config["run_config.evaluation.db_path"]
        prefix = "dependent_data_"
        suffix = "/memory/people.db"
        alpha = string.split(prefix)[-1].split(suffix)[0]
        res_config["alpha"] = float(alpha)

        # add meta data
        res |= res_config
        all_res.append(res)

        print(f"Task {task_id} done")

    df = pd.DataFrame(all_res)
    path = RESULT_PATH / gridname
    df.to_csv(path)


def get_data(gridname: str) -> pd.DataFrame:
    """Load data from csv file."""
    path = RESULT_PATH / gridname
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------


def save_plot(figname: str, format: str = "pdf", dpi: int = 100) -> None:
    """Save figure in pdf format."""
    if not FIGURE_PATH.exists():
        FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    save_dir = FIGURE_PATH / f"{figname}.{format}"
    plt.savefig(save_dir, format=format, bbox_inches="tight", dpi=dpi)


def plot_params_bound(
    df: pd.DataFrame,
    acc_threshold: float,
    figname: str,
    tool_label: str = "In-Tool",
    weight_label: str = "In-Weight",
    figsize: tuple = (WIDTH, HEIGHT),
    save: bool = True,
    ncol: int = 1,
    loc: str = "upper center",
    bbox_to_anchor: tuple = (0.4, 0.87),
    palette: list = None,
) -> None:
    """
    Plot the evolution of the number of parameters needed to reach
    a given level of accuracy when the number of facts to learn grows.
    """
    fig, ax = plt.subplots(figsize=figsize)
    lw = LINEWIDTH
    ms = MARKER_SIZE
    alpha = ALPHA_GRID
    data_threshold = 1024

    # Recover values
    x = df["data.nb_data"].unique()
    all_y_mins = []
    all_y_mins_2 = []

    for seed in df["data.seed"].unique():
        root_ind = (df["data.seed"] == seed) & (df["accuracy_best"] >= acc_threshold)

        # with tool
        ind = (df["data.key"] == "qatool") & root_ind
        y_min = []
        for nb_data in x:
            tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
            y_min.append(tmp.min())
        all_y_mins.append(y_min)

        # without tool
        ind = (df["data.key"] == "qa") & root_ind
        y_min = []
        for nb_data in x:
            tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
            y_min.append(tmp.min())
        all_y_mins_2.append(y_min)

    # Data threshold
    ax.axvline(x=data_threshold, linewidth=2, linestyle="--", color="black")

    # Recover the number of facts (each people has 4 attributes)
    nb_facts = 4 * x
    y_mean = np.nanmean(np.array(all_y_mins), axis=0)
    y_std = np.nanstd(np.array(all_y_mins), axis=0)
    ax.plot(
        nb_facts, y_mean, label=tool_label, linestyle="-", linewidth=lw, marker="o", markersize=ms, color=palette[0]
    )
    ax.fill_between(nb_facts, y_mean - y_std, y_mean + y_std, alpha=alpha, color=palette[0])

    y_mean = np.nanmean(np.array(all_y_mins_2), axis=0)
    y_std = np.nanstd(np.array(all_y_mins_2), axis=0)
    ax.plot(
        nb_facts, y_mean, label=weight_label, linestyle="-", linewidth=lw, marker="o", markersize=ms, color=palette[1]
    )
    ax.fill_between(nb_facts, y_mean - y_std, y_mean + y_std, alpha=alpha, color=palette[1])

    # Axis
    ax.set_ylabel(f"Model Size \n (s.t. Recall > {int(acc_threshold * 100)}%)")
    ax.set_xlabel("Dataset Size (#Facts)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10**1, 10**3, 10**5])
    ax.set_xticklabels([r"10$^\text{1}$", r"10$^\text{3}$", r"10$^\text{5}$"])
    ax.grid(alpha=0.6)
    ax.minorticks_off()

    # Set legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        frameon=False
    )

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


def plot_params_bound_recall(
    df: pd.DataFrame,
    figname: str,
    figsize: tuple = (WIDTH, HEIGHT),
    save: bool = True,
    ncol: int = 2,
    loc: str = "upper center",
    bbox_to_anchor: tuple = (0.56, 1.14),
    palette: list = None,
) -> None:
    """
    Plot the evolution of the number of parameters requirement with the factual recall accuracy for in-weight learning.
    """
    fig, ax = plt.subplots(figsize=figsize)
    lw = LINEWIDTH
    ms = MARKER_SIZE
    alpha = ALPHA_GRID

    # Recover values
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    nb_datas = [1024, 2048, 4096, 8192]
    labels = {1024: "4K facts", 2048: "8K facts", 4096: "16K facts", 8192: "32K facts"}
    key = "qa"

    # For better visualization (to avoid high values to dominate small ones)
    for i, nb_data in enumerate(nb_datas):
        all_y_mins = []
        for seed in df["data.seed"].unique():
            root_ind = (df["data.seed"] == seed) & (df["data.nb_data"] == nb_data)
            ind = (df["data.key"] == key) & root_ind
            y_min = []
            for acc in x:
                tmp = df[(df["accuracy_best"] >= acc) & ind]["nb_params"]
                y_min.append(tmp.min())
            all_y_mins.append(y_min)
        y_mean = np.nanmean(np.array(all_y_mins), axis=0)
        y_std = np.nanstd(np.array(all_y_mins), axis=0)
        ax.plot(
            x,
            y_mean,
            label=f"{labels[nb_data]}",
            linestyle="-",
            linewidth=lw,
            marker="o",
            markersize=ms,
            color=palette[i],
        )
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=palette[i])

        ax.set_ylabel("Model Size")
        ax.set_xlabel("Factual Recall Accuracy (%)")
        ax.set_xticks([0.05, 0.5, 0.95])
        ax.set_xticklabels([5, 50, 95])
        ax.set_yticks([50_000, 150_000, 250_000])
        ax.set_yticklabels(["50K", "150K", "250K"])
        ax.grid(alpha=0.6, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(direction="out", length=6)
        ax.minorticks_off()

    # Set legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
    )

    # Global
    sns.despine(fig, ax, trim=False, right=True, offset=10)

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


def plot_params_bound_recall_grouped(
    df: pd.DataFrame,
    figname: str,
    figsize: tuple = (WIDTH, HEIGHT),
    save: bool = True,
    ncol: int = 1,
    loc: str = "upper center",
    bbox_to_anchor: tuple = (0.15, 0.92),
    palette: list = None,
) -> None:
    """
    Plot the evolution of the number of parameters requirement with the factual
    recall accuracy for in-weight learning for various numbe of facts.
    """
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    nb_datas = np.asarray([[256, 512, 1024], [2048, 4096, 8192]])
    data_labels = {
        256: "1K facts",
        512: "2K facts",
        1024: "4K Facts",
        2048: "8K Facts",
        4096: "16K Facts",
        8192: "32K Facts",
    }
    nrows = len(nb_datas)
    ncols = len(nb_datas[0])
    figsize = (figsize[0] * ncols, figsize[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
    lw = LINEWIDTH
    ms = MARKER_SIZE
    alpha = ALPHA_GRID

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            nb_data = nb_datas[i, j]

            # Recover values
            labels = {"qatool": "In-Tool", "qa": "In-Weight"}
            keys = ["qatool", "qa"]

            # For better visualization (to avoid high values to dominate small ones)
            for k, key in enumerate(keys):
                all_y_mins = []
                for seed in df["data.seed"].unique():
                    root_ind = (df["data.seed"] == seed) & (df["data.nb_data"] == nb_data)
                    ind = (df["data.key"] == key) & root_ind
                    y_min = []
                    for acc in x:
                        tmp = df[(df["accuracy_best"] >= acc) & ind]["nb_params"]
                        y_min.append(tmp.min())
                    all_y_mins.append(y_min)
                y_mean = np.nanmean(np.array(all_y_mins), axis=0)
                y_std = np.nanstd(np.array(all_y_mins), axis=0)
                ax.plot(
                    x,
                    y_mean,
                    label=f"{labels[key]}",
                    linestyle="-",
                    linewidth=lw,
                    marker="o",
                    markersize=ms,
                    color=palette[k],
                )
                ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=palette[k])

                ax.set_ylabel("Model Size")
                ax.set_xlabel("Factual Recall Accuracy (%)")
                ax.set_xticks([0.05, 0.5, 0.95])
                ax.set_xticklabels([5, 50, 95])
                ax.set_yticks([50_000, 150_000, 250_000])
                ax.set_yticklabels(["50K", "150K", "250K"])
                ax.grid(alpha=0.6, lw=1.3)
                ax.spines["left"].set_linewidth(1)
                ax.spines["right"].set_linewidth(1)
                ax.spines["top"].set_linewidth(1)
                ax.spines["bottom"].set_linewidth(1)
                ax.tick_params(direction="out", length=6)
                ax.minorticks_off()
                ax.set_title(f"{data_labels[nb_data]}", fontsize=FONTSIZE)

    # Global
    sns.despine(fig, ax, trim=False, right=True, offset=10)

    # Set legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
        fontsize=FONTSIZE,
    )

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


def plot_params_accuracy(
    df: pd.DataFrame,
    nb_data: int,
    figname: str,
    figsize: tuple = (WIDTH, HEIGHT),
    save: bool = True,
    ncol: int = 1,
    loc: str = "upper center",
    bbox_to_anchor: tuple = (0.45, 0.92),
    palette: list = None,
) -> None:
    """
    Plot the evolution of the number of parameters requirement with the factual recall accuracy.
    """
    fig, ax = plt.subplots(figsize=figsize)
    lw = LINEWIDTH
    ms = MARKER_SIZE
    alpha = ALPHA_GRID

    # Recover values
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    labels = {"qatool": "In-Tool", "qa": "In-Weight"}
    keys = ["qatool", "qa"]

    # For better visualization (to avoid high values to dominate small ones)
    for i, key in enumerate(keys):
        all_y_mins = []
        for seed in df["data.seed"].unique():
            root_ind = (df["data.seed"] == seed) & (df["data.nb_data"] == nb_data)
            ind = (df["data.key"] == key) & root_ind
            y_min = []
            for acc in x:
                tmp = df[(df["accuracy_best"] >= acc) & ind]["nb_params"]
                y_min.append(tmp.min())
            all_y_mins.append(y_min)
        y_mean = np.nanmean(np.array(all_y_mins), axis=0)
        y_std = np.nanstd(np.array(all_y_mins), axis=0)
        ax.plot(
            x,
            y_mean,
            label=f"{labels[key]}",
            linestyle="-",
            linewidth=lw,
            marker="o",
            markersize=ms,
            color=palette[i],
        )
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha, color=palette[i])

        ax.set_ylabel("Model Size")
        ax.set_xlabel("Factual Recall Accuracy (%)")
        ax.set_xticks([0.05, 0.5, 0.95])
        ax.set_xticklabels([5, 50, 95])
        ax.set_yticks([50_000, 150_000, 250_000])
        ax.set_yticklabels(["50K", "150K", "250K"])
        ax.grid(alpha=0.6, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(direction="out", length=6)
        ax.minorticks_off()

    # Set legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
    )

    # Global
    sns.despine(fig, ax, trim=False, right=True, offset=10)

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


def plot_in_tool_generalization(
    df: pd.DataFrame,
    figname: str,
    figsize: tuple = (WIDTH, HEIGHT),
    save: bool = True,
    ncol: int = 1,
    loc: str = "upper center",
    palette: list = None,
    bbox_to_anchor: tuple = (0.78, 0.78),
) -> None:
    """
    Plot the evolution of the OOD accuracy with in-tool learning when the number of facts to learn grows.
    """

    # Set plotting parameters
    nb_params = [60372, 129752, 591488]
    labels_params = {60372: "60K", 129752: "130K", 591488: "600K"}

    # Number of facts at which the learning transition occurs (memorization to generalization)
    data_threshold = 1024

    # Accuracy of the best random baselines i.e., that outputs the fact the most present in the evaluation data
    best_acc_random = 0.1

    # Figure parameters
    fig, ax = plt.subplots(figsize=figsize)
    lw = LINEWIDTH
    ms = MARKER_SIZE

    # Data threshold and best random model
    ax.axvline(x=data_threshold, linewidth=2.5, linestyle="--", color="black")
    ax.axhline(y=best_acc_random, linewidth=2.5, label="best random", linestyle="--", color="red")

    # Recover values
    x = np.sort(df["data.nb_data"].unique())
    key = "qatool"
    for i, model_size in enumerate(nb_params):
        # with tool
        ind = (df["data.key"] == key) & (df["nb_params"] == model_size)
        acc = []
        for nb_data in x:
            tmp = df[(df["data.nb_data"] == nb_data) & ind]["accuracy_best"]
            acc.append(tmp.values[0])

        acc = np.asarray(acc)

        # Recover the number of facts (each people has 4 attributes)
        nb_facts = 4 * x

        ax.plot(
            nb_facts,
            acc,
            label=f"{labels_params[model_size]} params",
            linestyle="-",
            marker="o",
            markersize=ms,
            linewidth=lw,
            color=palette[i],
        )

        # metadata
        ax.set_ylabel("OOD Accuracy (%)")
        ax.set_xlabel("Dataset Size (#Facts)")
        ax.set_xscale("log")
        ax.set_xticks([10**2, 10**3, 10**4, 10**5])
        ax.set_xticklabels([r"10$^\text{2}$", r"10$^\text{3}$", r"10$^\text{4}$", r"10$^\text{5}$"])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 50, 100])
        ax.grid(alpha=0.6, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(direction="out", length=6)
        ax.minorticks_off()

    # Set legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines[::-1], labels[::-1]
    fig.legend(
        lines, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, fancybox=True, borderaxespad=0, ncol=ncol, fontsize=8
    )

    # Global
    sns.despine(fig, ax, trim=False, right=True, offset=10)

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


def plot_compressibility(
    figname: str,
    figsize: tuple = (WIDTH, HEIGHT),
    save: bool = True,
    ncol: int = 2,
    loc: str = "upper center",
    palette: list = None,
    bbox_to_anchor: tuple = (0.6, 1.1), #(0.45, 0.7),
) -> None:
    """
    Plot the evolution of parameters requirements to obtain a recall of 100% when the attributes
    become more and more independent.
    """

    fig, ax = plt.subplots(figsize=figsize) 
    lw = LINEWIDTH
    ms = MARKER_SIZE

    # Recover values
    alphas = np.asarray([0, 0.25, 0.5, 0.75, 1])

    labels = {8192: "32K facts", 1000: "4K facts"}
    for i, nb_data in enumerate([8192, 1000]):
        # Get grid
        gridname = f"grid_dependent_{nb_data}.csv"
        df = get_data(gridname)

        # Recover data
        y_mins = []
        for alpha in alphas:
            root_ind = (df["alpha"] == 1 - alpha) & (df["accuracy_best"] == 1)

            # without tool
            ind = (df["data.key"] == "qa") & root_ind
            tmp = df[(df["data.nb_data"] == nb_data) & ind]["nb_params"]
            y_mins.append(tmp.min())

        # Increase range of second curve for visualization (given the differences in range, it would collapse otherwise)
        if i > 0:
            y_mins = 2 * np.asarray(y_mins)

        # Plot evolution
        ax.plot(
            alphas,
            y_mins,
            linestyle="-",
            label=labels[nb_data],
            linewidth=lw,
            marker="o",
            markersize=ms,
            color=palette[i],
        )

        # metadata
        ax.grid(alpha=0.6, lw=1.3)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.minorticks_off()

    # Adapt axes with legends and to deal with the change of range between 8192 and 1000 facts
    ax.set_xlabel(r"Correlation $\alpha$")
    ax.set_ylabel("Model Size")
    ax.locator_params(axis="x", nbins=3)
    ax.set_yticks([50_000, 125_000, 162_500, 200_000, 275_000])
    ax.set_yticklabels(["25K", "75K", r"$\vdots$ ", "200K", "275K"])


    # Global
    sns.despine(fig, ax, trim=False, right=True, offset=10)

    # Remove useless grid lines and ticks
    line = ax.get_ygridlines()[2]
    line.set_visible(False)
    ax.yaxis.majorTicks[2].tick1line.set_markersize(0)

    # Set legend
    lines_labels = [fig.axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fancybox=True,
        borderaxespad=0,
        ncol=ncol,
    )

    # Show plot
    plt.tight_layout()
    if save:
        save_plot(figname=figname)
    plt.show()


# ------------------------------------------------------------------------------
# Controlled experiments
# ------------------------------------------------------------------------------


def recover_ood_eval(gridname: str = "grid_ood.csv") -> None:
    """Recover OOD accuracy of models (see apps/memory/config/ood_grid.yaml)."""
    dirname = "memory_ood_grid"
    save_data_from_grid(dirname=dirname, gridname=gridname)


def recover_compressibility_eval(gridname: str = "grid_dependent_8192.csv") -> None:
    """Recover accuracy of models on compressed data (see apps/memory/config/compressibility/run_1.yaml)."""
    save_compressed_data_config(gridname=gridname)


def bounds(acc_threshold: float = 0.95) -> None:
    """Verify theoretical bounds on the number of parameters."""
    gridname = "grid4.csv"
    df = get_data(gridname)
    palette = ["#5e6d9e", "#c2a7da"]
    plot_params_bound(df=df, acc_threshold=acc_threshold, figname=f"parameter_bounds_{acc_threshold}", palette=palette)


def bounds_grouped() -> None:
    """Verify theoretical bounds on the number of parameters."""
    gridname = "grid4.csv"
    df = get_data(gridname)
    palette = ["#5e6d9e", "#c2a7da"]
    plot_params_bound_recall_grouped(df=df, figname="parameter_bounds_grouped", palette=palette)


def bounds_recall() -> None:
    """Verify evolution of parameters requirements with factual recall accuracy for in-weight learning."""
    gridname = "grid4.csv"
    df = get_data(gridname)
    palette = ["#e3d0ef", "#d0a2cf", "#c2a7da", "#825c98"]

    plot_params_bound_recall(df=df, figname="parameter_bounds_recall", palette=palette)


def params_acc(nb_data: int = 8192) -> None:
    """Verify evolution of parameters requirements with factual recall accuracy."""
    gridname = "grid4.csv"
    df = get_data(gridname)
    palette = ["#5e6d9e", "#c2a7da"]
    plot_params_accuracy(df=df, nb_data=nb_data, figname=f"parameter_accuracy_{nb_data}", palette=palette)

def generalization(figname: str = "in_tool_generalization") -> None:
    """Verify benefits of in-tool learning for generalization."""
    gridname = "grid_ood.csv"
    df = get_data(gridname)
    palette = ["#ccdcef", "#a2b8da", "#5e6d9e"]
    plot_in_tool_generalization(df=df, figname=figname, palette=palette)


def compressibility(figname: str = "compressibility") -> None:
    """Verify compressibility of parameters requirements with dependent attributes."""
    palette = ["#9d82bf", "#e3d0ef"]
    plot_compressibility(figname=figname, palette=palette)


# %% Main
def main() -> None:
    r"""
    Visualize the analysis experiments from a configuration file specified by cli argument.

    Usage:
    To plot the empirical verification of the bounds with accuracy threshold of 0.95
    ```shell
    python -m apps.memory.plots.analysis bounds 0.95
    ```
    To plot the empirical verification of in-tool generalization
    ```shell
    python -m apps.memory.plots.analysis gen
    ```
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "ood_eval": recover_ood_eval,
            "comp_eval": recover_compressibility_eval,
            "bounds": bounds,
            "group": bounds_grouped,
            "rec": bounds_recall,
            "gen": generalization,
            "comp": compressibility,
            "acc": params_acc,
        }
    )


# %% CLI
if __name__ == "__main__":
    main()
# %%