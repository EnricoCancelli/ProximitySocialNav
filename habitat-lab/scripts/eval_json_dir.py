import argparse
from collections import defaultdict
import glob
import numpy as np
from os import path as osp
import tqdm

from parse_jsons import get_best_ckpt

parser = argparse.ArgumentParser()
parser.add_argument("json_dir", help="path to dir containing social/pointnav")
args = parser.parse_args()


def get_dirs(dirpath, regex="*"):
    return list(
        filter(lambda x: osp.isdir(x), glob.glob(osp.join(dirpath, regex)))
    )


best_data = defaultdict(list)
json_queue = []

for nav_dir in get_dirs(args.json_dir):
    for seed in range(1, 4):
        for json_dir in get_dirs(nav_dir, regex=f"*seed{seed}*"):
            jkey = (
                f"{osp.basename(nav_dir)}_"
                f"{osp.basename(json_dir).split('seed')[0][:-1]}"
            )
            json_queue.append((json_dir, jkey))

for json_dir, jkey in tqdm.tqdm(json_queue):
    id_succ = get_best_ckpt(
        json_dir,
        silent=True,
        max_ckpt_id=100,
        val_split="val",
        test_split="test",
    )
    best_data[jkey].append(id_succ)

# Make sure all have 3 seeds
means_std = {}
for k, v in best_data.items():
    assert len(v) == 3, f"{k} has length {len(v)}, not 3!"

    succ = np.array(list(map(lambda x: x[1], v)))
    succ_mean = np.mean(succ)
    succ_std = np.std(succ)
    means_std[k] = (succ_mean, succ_std, list(map(lambda x: x[0], v)))

sorted_m_s = sorted(means_std.keys(), key=lambda x: -means_std[x][0])

for i in sorted_m_s:
    print(
        f"{i}:\t{means_std[i][0]*100:.2f},\t{means_std[i][1]*100:.2f},\t{means_std[i][2]}"
    )
