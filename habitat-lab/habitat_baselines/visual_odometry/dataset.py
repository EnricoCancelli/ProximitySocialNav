import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose
from habitat_baselines.visual_odometry.augmentation import Flip, Swap
import json
import yaml
import os
import numpy as np
import quaternion
import matplotlib.pyplot as plt

aug_dict = {
    "Flip": Flip,
    "Swap": Swap
}


class VODataset(Dataset):
    def __init__(self, path, dim, rgb=True, depth=True, aug=None):
        self.path = path
        self.dim = dim
        self.rgb = rgb
        self.depth = depth
        self.aug = aug

        assert os.path.exists(self.path), "path does not exists"
        assert os.path.exists(os.path.join(self.path, "entries.json")),\
            "entries file does not exists"
        with open(os.path.join(self.path, "entries.json"), "r") as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def calculate_deltas(pos0, pos1):
        pos_s, rot_s = pos0[0:3].numpy(), pos0[3:].numpy()
        pos_t, rot_t = pos1[0:3].numpy(), pos1[3:].numpy()

        rot_s, rot_t = (
            quaternion.as_quat_array(rot_s),
            quaternion.as_quat_array(rot_t)
        )

        rot_s2w, rot_t2w = (
            quaternion.as_rotation_matrix(rot_s),
            quaternion.as_rotation_matrix(rot_t)
        )

        trans_s2w, trans_t2w = (
            np.zeros(shape=(4, 4), dtype=np.float32),
            np.zeros(shape=(4, 4), dtype=np.float32)
        )
        trans_s2w[3, 3], trans_t2w[3, 3] = 1., 1.
        trans_s2w[0:3, 0:3], trans_t2w[0:3, 0:3] = rot_s2w, rot_t2w
        trans_s2w[0:3, 3], trans_t2w[0:3, 3] = pos_s, pos_t

        '''
            Construct the 4x4 transformation [world-->agent] matrices
            corresponding to the source and target agent state
            by inverting the earlier transformation matrices
        '''
        trans_w2s = np.linalg.inv(trans_s2w)
        trans_w2t = np.linalg.inv(trans_t2w)

        '''
            Construct the 4x4 transformation [target-->source] matrix
            (viewing things from the ref frame of source)
            -- take a point in the agent's coordinate at target state,
            -- transform that to world coordinates (trans_t2w)
            -- transform that to the agent's coordinates at source state (trans_w2s)
        '''
        trans_t2s = np.matmul(trans_w2s, trans_t2w)

        rotation = quaternion.as_rotation_vector(
            quaternion.from_rotation_matrix(trans_t2s[0:3, 0:3])
        )
        assert np.abs(rotation[0]) < 1e-05
        assert np.abs(rotation[2]) < 1e-05

        return {
            "translation": trans_t2s[0:3, 3],
            "rotation": rotation[1] if np.abs(rotation[1]) < np.pi else rotation[1] - 2*np.pi
        }

    def __getitem__(self, index) -> T_co:
        entry = self.entries[index]
        rgbd_0 = torch.load(os.path.join(self.path, "rgbd_{}.pt".format(entry[0])), map_location="cpu")
        rgbd_1 = torch.load(
            os.path.join(self.path, "rgbd_{}.pt".format(entry[1])), map_location="cpu")

        rgbd_0[:, :, :-1] = rgbd_0[:, :, :-1] / 255.
        rgbd_1[:, :, :-1] = rgbd_1[:, :, :-1] / 255.
        if not self.rgb:
            rgbd_0 = rgbd_0[:, :, -1]
            rgbd_1 = rgbd_1[:, :, -1]
        if not self.depth:
            rgbd_0 = rgbd_0[:, :, :-1]
            rgbd_1 = rgbd_1[:, :, :-1]
        vis = torch.cat([rgbd_0, rgbd_1], dim=2).permute(2,0,1)
        pa = torch.load(os.path.join(self.path, "pa_{}.pt".format(entry[1])), map_location="cpu")
        pos_0 = torch.load(os.path.join(self.path, "gs_{}.pt".format(entry[0])), map_location="cpu")
        pos_1 = torch.load(os.path.join(self.path, "gs_{}.pt".format(entry[1])), map_location="cpu")
        delta = VODataset.calculate_deltas(pos_0, pos_1)
        assert np.sign(delta["rotation"]) == np.sign(pa[1].item())
        #assert np.sign(delta["translation"][-1]) != np.sign(pa[0].item()) or np.abs(delta["translation"][-1]) < 0.07
        if self.aug is not None:
            for aug in self.aug:
                if random.random() >= aug.p:
                    vis, pa, delta = aug(vis, pa, delta)
        return vis, pa, delta

    @classmethod
    def from_config_path(cls, f_path):
        with open(f_path, "r") as f:
            config = yaml.load(f, yaml.loader.FullLoader)

        config = config["data"]
        aug = [aug_dict[c](config["p"]) for c in config["aug"]]
        return cls(os.path.join(config["path"], config["split"]), config["dim"], config["rgb"], config["depth"], aug)


if __name__ == "__main__":
    #data = VODataset("../../vodata/train", dim=3)
    data = VODataset.from_config_path("./config_base.yaml")
    import tqdm
    for i in tqdm.tqdm(range(len(data))):
        data[i]
