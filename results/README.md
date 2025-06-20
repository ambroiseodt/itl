#### Memory results

The results `grid{i}.csv`, for various `i`, were obtained by running a job-array configured in `apps/memory/config/grid.yaml`.
The first one was run with `10_000` steps, with evals every `20` steps, and the second one with `100_000` steps and a single evals at the end of training.
The third one was run with `100_000` steps, single eval yet a finer grid of parameters for one seed and the second one similarly for `10` seeds (see `apps/memory/config/fine_grid.yaml`).
The results `grid_ood.csv` comes from OOD evaluation with varying number of facts to learn and model sizes (see `apps/memory/config/ood_grid.yaml`).