import argparse
import glob
from os import path as osp

parser = argparse.ArgumentParser()
parser.add_argument("json_dir", help="path to dir containing social/pointnav")
args = parser.parse_args()


def get_dirs(dirpath, regex="*"):
    return list(
        filter(lambda x: osp.isdir(x), glob.glob(osp.join(dirpath, regex)))
    )


json_queue = []

for nav_dir in get_dirs(args.json_dir):
    for json_dir in get_dirs(nav_dir):
        print(json_dir, len(glob.glob(osp.join(json_dir, '*.json'))))
