#### Memory results

The results `grid{i}.csv`, for various `i`, were obtained by running a job-array configured in `apps/memory/config/grid.yaml`. 
The first one was run with `10_000` steps, with evals every `20` steps, and the second one with `100_000` steps and a single evals at the end of training.
The third one was run with `100_000` steps, single eval yet a finer grid of parameters (see `apps/memory/config/fine_grid.yaml`).