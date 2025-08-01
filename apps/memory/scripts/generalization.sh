#!/usr/bin/bash

# This file is useful to show generalization with in-tool learning by training
# models and evaluating them on ood data. To that end, run the following command.
# It will create a dedicated tmux session to launch the grid of experiments.
# To launch experiments with Slurm, replace python -m apps.memory.local_grid
# by python -m nanollama.launcher.
# ```shell
# $ bash <path_to_file_folder>/generalization.sh
# ```

tmux new-session -d -s OODGRID
tmux send-keys -t OODGRID "python -m apps.memory.local_grid apps/memory/config/ood_grid.yaml" C-m