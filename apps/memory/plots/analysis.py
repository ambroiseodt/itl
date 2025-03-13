# %%
from logging import getLogger
from pathlib import PosixPath

import pandas as pd

from nanollama.visualization.loader import get_config_info, get_task_ids, load_jsonl_to_numpy

logger = getLogger("nanollama")


# %%

steps = [20, 100, 1000, 10000]

log_dir = PosixPath.home() / "grid_new"
task_id = 1
config_keys = ["data.n_data", "data.key", "model.emb_dim", "model.nb_layers", "model.block.nb_heads"]

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
            idx = (step == snapshot).argmax()
            res[f"loss_{snapshot}"] = data["loss"][idx]
            res[f"accuracy_{snapshot}"] = data["accuracy"][idx + 1]
        except Exception as e:
            logger.error(f"Error extracting loss for snapshot {snapshot}: {str(e)}")

    # add meta data
    res |= get_config_info(log_dir, task_id, keys=config_keys)
    all_res.append(res)

    print(f"Task {task_id} done")

df = pd.DataFrame(all_res)

# %%

