#!/usr/bin/bash

# This file is useful to empirically verify in-weight and in-tool learning parameters bounds.
# To that end, run the following command. It will create a dedicated tmux session to launch
# the grid of experiments. To launch experiments with Slurm, replace python -m apps.memory.local_grid
# by python -m nanollama.launcher.
# ```shell
# $ bash <path_to_file_folder>/params_bound.sh
# ```

tmux new-session -d -s BOUND
tmux send-keys -t BOUND "python -m apps.memory.local_grid apps/memory/config/fine_grid.yaml" C-m