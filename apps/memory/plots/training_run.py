"""
Generate Plots to visualize training runs
"""

# %% Imports
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nanollama.visualization.loader import load_results

# %% Load results

# log_dir = Path.home() / "memory"
log_dir = Path.home() / "grid_first_runs"
task_id = "1"
metric_dir = log_dir / "metrics" / task_id

dfs = {}
dfs["train"] = load_results(metric_dir, "raw")
dfs["profiler"] = load_results(metric_dir, "prof")
dfs["eval"] = load_results(metric_dir, "evas")

# %% Plot some of them with Plotly

# should the metric to plot
metrics = ["train.loss", "train.ts", "eval.loss"]

# create a figure with subplots (you can add arguments such as vspace, ...)
n_cols = 2
n_rows = (len(metrics) + 1) // n_cols

fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=metrics)

# Add traces for each column
for i, metric in enumerate(metrics):
    src, metric = metric.split(".")
    df = dfs[src][["step", metric]]
    row = i // n_cols + 1
    col = i % n_cols + 1
    ax = fig.add_trace(go.Scatter(x=df["step"], y=df[metric], mode="lines", name=metric), row=row, col=col)
    fig.update_xaxes(title_text="step", row=row, col=col)
    fig.update_yaxes(title_text=metric, row=row, col=col, type="log")
# Update layout
fig.update_layout(showlegend=False, height=300 * n_rows, width=600 * n_cols)
fig.show()


#
# %%
