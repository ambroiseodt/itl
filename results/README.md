In this folder, we provde the results files needed to reproduce the figures of our [paper](https://arxiv.org/pdf/2508.20755).

#### Memory results of Section 5
The results `grid4.csv` was obtained by running a job-array configured in `apps/memory/config/fine_grid.yaml` with `100_000` steps and `10` seeds (see `apps/memory/config/fine_grid.yaml`).
The results `grid_ood.csv` comes from OOD evaluation with varying number of facts to learn and model sizes (see `apps/memory/config/ood_grid.yaml`).
The results `grid_dependent_1000.csv` and `grid_dependent_8192.csv` comes from compressibility evaluation with varying number of facts to learn and model sizes (see `apps/memory/config/compressibility/`).

#### Large-scale results of Section 6
The results `large_scale_results.csv` comes from the large scale finetuning experiments of apps/large_scale on Llama and SmolLM models.