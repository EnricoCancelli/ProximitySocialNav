import numpy as np
import quaternion
import habitat_sim
from habitat.tasks.nav.nav import PointGoalSensor, NavigationEpisode
from habitat.config import Config
from habitat.core.simulator import Simulator
from habitat.core.registry import registry
from habitat_baselines.visual_odometry.model import VOModel
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector
from habitat_baselines.common.obs_transformers import ResizeShortestEdge, ResizeSquare

from typing import Any
import torch


@registry.register_sensor
class OdometricGpsCompassSensor(PointGoalSensor):
    cls_uuid = 'odometric_gpscompass'

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self.pointgoal = None
        self.prev_observations = None

        device = torch.device('cuda',
                              sim.habitat_config.HABITAT_SIM_V0.GPU_DEVICE_ID)

        self.vo_model = VOModel.from_config_path(config.VO_PATH).to(device)

        checkpoint = torch.load(config.VO_CKPT, map_location=device)
        self.vo_model.load_state_dict(checkpoint)
        self.vo_model.eval()
        self.device = device

        self.trans = ResizeSquare(size=224)  #temporary maybe

        super().__init__(sim=sim, config=config)

    def compute_pointgoal_after_offset(self, offset):
        #offset position and rotation, then compute new point nav
        x, y, z, angle = offset[:, 0].item(), offset[:, 1].item(), offset[:, 2].item(), offset[:, 3].item()
        position_offset = np.array([x, y, z], dtype=np.float32)
        rotation_offset = quaternion.as_rotation_matrix(
            habitat_sim.utils.quat_from_angle_axis(theta=angle, axis=np.asarray([0.,1.,0.]))
        )
        trans_c2s = np.zeros((4, 4), dtype=np.float32)
        trans_c2s[3, 3] = 1.
        trans_c2s[:3, 3] = position_offset
        trans_c2s[:3, :3] = rotation_offset

        return np.dot(
            np.linalg.inv(trans_c2s),  # noisy_T_prev2curr_state
            np.concatenate(
                (self.pointgoal, np.asarray([1.], dtype=np.float32)), axis=0)
        )

    def get_observation(self, observations, episode: NavigationEpisode,
                        *args: Any, **kwargs: Any):
        observations = self.trans(observations)
        episode_reset = 'action' not in kwargs
        episode_end = (not episode_reset) and (kwargs['action']['action'] == 0)

        if episode_reset:
            # at episode reset compute pointgoal and reset pointgoal_estimator
            source_position = np.array(episode.start_position,
                                       dtype=np.float32)
            source_rotation = quaternion_from_coeff(episode.start_rotation)
            goal_position = np.array(episode.goals[0].position,
                                     dtype=np.float32)

            direction_vector = goal_position - source_position

            direction_vector_agent = quaternion_rotate_vector(
                source_rotation.inverse(), direction_vector
            )

        elif not episode_end:
            inputs = []
            if "rgb" in observations:
                inputs.append(self.prev_observations["rgb"] / 255.)
            if "depth" in observations:
                inputs.append(self.prev_observations["depth"])
            if "rgb" in observations:
                inputs.append(observations["rgb"] / 255.)
            if "depth" in observations:
                inputs.append(observations["depth"])
            vis = torch.from_numpy(np.expand_dims(np.concatenate(inputs, axis=-1), axis=0)).to(self.device)
            vis = torch.permute(vis, (0, 3, 1, 2))
            pa = torch.as_tensor([[kwargs["action"]["action_args"]["lin_vel"], kwargs["action"]["action_args"]["ang_vel"]]], device=self.device)
            offset = self.vo_model(vis, pa)

            direction_vector_agent = self.compute_pointgoal_after_offset(offset)
        else:
            return

        self.pointgoal = direction_vector_agent[:3]
        self.prev_observations = observations

        if self._goal_format == 'POLAR':
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

        return direction_vector_agent



