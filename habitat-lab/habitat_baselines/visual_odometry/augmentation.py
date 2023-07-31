import numpy as np
import torch as tc


class Flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vis, pa, delta, *args, **kwargs):
        vis = tc.from_numpy(np.ascontiguousarray(vis.numpy()[:, ::-1, ...]))
        pa[1] *= -1
        delta["translation"][0] *= -1
        delta["rotation"] *= -1
        return vis, pa, delta


class Swap:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vis, pa, delta, *args, **kwargs):
        half_d = vis.shape[0]//2
        vis[0:half_d, ...], vis[half_d:, ...] = vis[half_d:, ...], vis[0:half_d, ...]
        pa *= -1
        delta["translation"] *= -1
        delta["rotation"] *= -1
        return vis, pa, delta
