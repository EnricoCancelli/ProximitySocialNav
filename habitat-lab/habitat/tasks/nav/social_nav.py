from typing import Any, Dict

import numpy as np

from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator, Sensor, SensorTypes
from habitat.tasks.nav.nav import NavigationTask, TopDownMap, Success
from habitat.utils.visualizations import maps

from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
    get_heading_error,
    quat_to_rad,
)

import math
import json
import yaml
import matplotlib.pyplot as plt
from array2gif import write_gif
from PIL import Image

import os

from habitat.tasks.utils import cartesian_to_polar
from copy import deepcopy

from habitat_sim.utils import gfx_replay_utils as gfx_replay

# evaluation sensors
def ccw(A, B, C):
    return (C[2] - A[2]) * (B[0] - A[0]) > (B[2] - A[2]) * (C[0] - A[0])


@registry.register_sensor
class FrontalSensor(Sensor):
    r"""
    """
    cls_uuid: str = "frontal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self.radius = config.RADIUS
        self.facing_steps = config.FACING_STEPS
        self.blinded_steps = config.BLINDED_STEPS
        self.slack = config.SLACK
        self.name = config.NAME
        self.ckp = config.CKP
        self.encounters = [None] * (3 * self._sim.num_people)

        assert os.path.isfile("ckp_cache.yml")
        with open("ckp_cache.yml") as f:
            self.current_ckp = yaml.load(f, Loader=yaml.FullLoader)["ckp_num"]

        assert 'semantic' in self._sim.sensor_suite.sensors
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        num = spaces.Dict({
            "num": spaces.Box(low=0, high=int(3 * self._sim.num_people), shape=(1,),
                       dtype=np.int32),
            "pid": spaces.Box(low=-1, high=int(3 * self._sim.num_people), shape=(1,),
                       dtype=np.int32)
        })
        return spaces.Dict({
                "frontal": num,
                "intersection": num,
                "blind_c": num,
                "others": num,
                "l_vel": spaces.Box(low=-1, high=+1, shape=(1,), dtype=np.float32)
            })

    @staticmethod
    def intersect(p1, p2, q1, q2):
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(
            p1, p2, q2)

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def is_facing(self, pos):
        a_head = -self.get_polar_angle() + np.pi / 2
        a_pos = self._sim.get_agent_state().position

        theta = math.atan2(pos[2] - a_pos[2], pos[0] - a_pos[0])
        theta = get_heading_error(a_head, theta)
        theta = theta if theta > 0 else 2 * np.pi + theta

        return theta < 1 / 3 * np.pi or theta > 5 / 3 * np.pi

    def get_observation(self, observations, *args: Any, **kwargs: Any) -> Any:
        base_c = {
            "num": 0,
            "pid": -1
        }
        comp_enc = {
            "frontal": deepcopy(base_c),
            "intersection": deepcopy(base_c),
            "blind_c": deepcopy(base_c),
            "others": deepcopy(base_c),
            "l_vel": kwargs["action"]["action_args"]["lin_vel"] \
                            if "action" in kwargs.keys() else 0.
        }

        a_pos = self._sim.get_agent_state().position
        # assert "action" not in kwargs.keys()

        for i, p in enumerate(self._sim.people):
            pos = p.current_position
            geo = self._sim.geodesic_distance(a_pos, pos)
            euc = np.sqrt((pos[2] - a_pos[2]) ** 2 + (pos[0] - a_pos[
                0]) ** 2)
            dist = geo if geo != float('inf') else euc
            if dist < self.radius and kwargs[
                "task"]._check_episode_is_active():
                # in encounter, check if already in or not
                facing = self.is_facing(pos)
                seeing = (observations["semantic"] == 600 + i).any()
                # init encounter
                if self.encounters[i] is None and facing:
                    self.encounters[i] = {
                        "n_steps": 1,
                        "pid": i,
                        "lin_vel": [kwargs["action"]["action_args"]["lin_vel"] \
                            if "action" in kwargs.keys() else 0],
                        "dist": [euc],
                        "initial_a_pos": a_pos,
                        "initial_p_pos": pos,
                        "facing": [facing],
                        "blind": [not seeing],
                        "obstacled": abs(dist - euc) > 0.5,  # 0.75
                        "type": None
                    }
                    if True:
                        self.encounters[i]["rgb"] = [observations["rgb"]/255.]
                        self.encounters[i]["semantic"] = [(observations["semantic"] == 600 + i).astype(np.float)]
                    print_cose = True
                elif self.encounters[i] is not None and (
                    self.encounters[i]["n_steps"] >= self.facing_steps or (
                    facing and self.encounters[i][
                    "n_steps"] < self.facing_steps)
                ):
                    # continue encounter
                    self.encounters[i]["n_steps"] += 1
                    self.encounters[i]["lin_vel"].append(
                    kwargs["action"]["action_args"][
                        "lin_vel"] \
                        if "action" in kwargs.keys() else 0)
                    self.encounters[i]["dist"].append(euc)
                    self.encounters[i]["facing"].append(facing)
                    self.encounters[i]["blind"].append(not seeing)
                    if True:
                        self.encounters[i]["rgb"].append(observations["rgb"]/255.)
                        self.encounters[i]["semantic"].append((observations["semantic"] == 600 + i).astype(np.float))

                    if self.encounters[i]["type"] is None and self.encounters[i]["n_steps"] >= self.facing_steps:
                        # early classification
                        #is an episode for sure
                        self.encounters[i]["type"] = "others"

                else:
                    self.encounters[i] = None

            elif self.encounters[i] is not None:
                # close encounter
                if self.encounters[i]["n_steps"] > 20:
                    # valid
                    theta_p = math.atan2(
                        self.encounters[i]["initial_p_pos"][2] - pos[2],
                        self.encounters[i]["initial_p_pos"][0] - pos[0]
                    )
                    theta_a = math.atan2(
                        self.encounters[i]["initial_a_pos"][2] - a_pos[2],
                        self.encounters[i]["initial_a_pos"][0] - a_pos[0]
                    )
                    theta_p = theta_p if theta_p > 0 else 2 * np.pi + theta_p
                    theta_a = theta_a if theta_a > 0 else 2 * np.pi + theta_a

                    p_opposite = abs(theta_p - theta_a) >= np.pi * (
                            1 - self.slack) and abs(
                        theta_p - theta_a) <= np.pi * (1 + self.slack)
                    p_same = abs(theta_p - theta_a) >= np.pi * (
                        - self.slack) and abs(
                        theta_p - theta_a) <= np.pi * (+ self.slack)
                    perpendicular = (abs(theta_p - theta_a) >= np.pi * (
                            0.5 - self.slack) and (
                                         theta_p - theta_a) <= np.pi * (
                                             0.5 + self.slack)) or (
                                        (abs(theta_p - theta_a) >= np.pi * (
                                                1.5 - self.slack) and (
                                             theta_p - theta_a) <= np.pi * (
                                                 1.5 + self.slack))
                                    )
                    blinded = all(
                        self.encounters[i]["blind"][0:self.blinded_steps])
                    type = None
                    if blinded and not all(self.encounters[i]["blind"]) and \
                        self.encounters[i]["obstacled"]:  # and not p_same:
                        type = "blind_c"
                    elif perpendicular and not blinded and \
                        FrontalSensor.intersect(
                            self.encounters[i]["initial_a_pos"],
                            a_pos,
                            self.encounters[i]["initial_p_pos"],
                            pos
                        ):
                        type = "intersection"
                    elif p_opposite and not blinded:
                        type = "frontal"
                    elif p_same and not blinded:
                        type = "follow"
                    else:
                        type = "others"

                    if type is not None:
                        directory = self.name
                        print(str(self.current_ckp), str(self.ckp))
                        if str(self.current_ckp) == str(self.ckp):
                            if not os.path.exists(directory):
                                os.mkdir(directory)

                            color = np.tile([1., 0., 0.], observations["depth"].shape)

                            imgs = [
                                Image.fromarray(((img*0.7+color*sem*0.3)*255).astype(np.uint8))
                                for img, sem in
                                    zip(self.encounters[i]["rgb"], self.encounters[i]["semantic"])
                            ]
                            #print("eeeeeeeeeeeeee")
                            imgs[0].save(directory+"/array_{}_{}_{}_{}_{}.gif".format(type,
                                                                       kwargs[
                                                                           "episode"].scene_id.split(
                                                                           "/")[
                                                                           -1],
                                                                       kwargs[
                                                                           "episode"].episode_id,
                                                                       self.encounters[
                                                                           i][
                                                                           "n_steps"],
                                                                       i),
                                     save_all=True,
                                     append_images=imgs[1:], duration=100,
                                     loop=0)
                            # put metrics
                            file_name = "array_{}_{}_{}_{}_{}.txt".format(type,
                                                                      kwargs[
                                                                          "episode"].scene_id.split(
                                                                          "/")[
                                                                          -1],
                                                                      kwargs[
                                                                          "episode"].episode_id,
                                                                      self.encounters[
                                                                          i][
                                                                          "n_steps"],
                                                                      i)
                            with open(os.path.join(directory, file_name), "w") as p:
                                json.dump({"dist": self.encounters[i]["dist"],
                                       "lin_vel": self.encounters[i]["lin_vel"],
                                       "failed": kwargs["task"].is_stop_called and kwargs["task"].measurements.measures["human_collision"].get_metric()}, p)

                self.encounters[i] = None

            if self.encounters[i] and self.encounters[i]["type"] is not None:
                comp_enc[self.encounters[i]["type"]]["num"] += 1
                comp_enc[self.encounters[i]["type"]]["pid"] = i

        return {k: ({sk: np.array([sv]) for sk, sv in v.items()} if k !="l_vel" else np.array([v])) for k, v in comp_enc.items()}
        #np.array([0], dtype=np.float32)

    def get_observation_ru(self, *args: Any, **kwargs: Any) -> Any:
        a_pos = self._sim.get_agent_state().position
        a_head = self._sim.get_agent_state().rotation  # 2*np.arccos(self._sim.get_agent_state().rotation.w)

        a_head = -self.get_polar_angle() + np.pi / 2  # -quat_to_rad(a_head) + np.pi / 2

        angles = [0] * self.num_bins
        for person in self._sim.people:
            pos = person.current_position
            theta = math.atan2(pos[2] - a_pos[2], pos[0] - a_pos[0])
            theta = get_heading_error(a_head, theta)
            theta = theta if theta > 0 else 2 * np.pi + theta

            bin = int(theta / (2 * np.pi / self.num_bins))

            dist = np.sqrt((pos[2] - a_pos[2]) ** 2 + (pos[0] - a_pos[
                0]) ** 2)  # self._sim.geodesic_distance(a_pos, pos)
            norm_dist = max(1 - dist / self.thres, 0)
            if norm_dist > angles[bin]:
                angles[bin] = norm_dist

        return np.array(angles, dtype=np.float32)


@registry.register_sensor
class RiskSensor(Sensor):
    r"""Sensor for observing social risk to which the agent is subjected".

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "risk"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self.thres: float = config.THRES
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        a_pos = self._sim.get_agent_state().position
        distances = []
        for person_id in self._sim.get_existing_object_ids():
            pos = self._sim.get_translation(person_id)
            dist = self._sim.geodesic_distance(a_pos, pos)  # , episode)
            distances.append(dist)

        return np.array([max(1 - min(distances) / self.thres, 0)],
                        dtype=np.float32) if len(distances) != 0 else np.array([0], dtype=np.float32)


@registry.register_sensor
class SocialCompassSensor(Sensor):
    r"""
    Implementation of people relative position sensor
    """

    cls_uuid: str = "social_compass"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        # parameters
        self.thres = config.THRES
        self.num_bins = config.BINS
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high="inf", shape=(self.num_bins,),
                          dtype=np.float32)

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def get_observation(self, *args: Any, **kwargs: Any) -> Any:
        a_pos = self._sim.get_agent_state().position
        a_head = self._sim.get_agent_state().rotation  # 2*np.arccos(self._sim.get_agent_state().rotation.w)

        a_head = -self.get_polar_angle() + np.pi / 2  # -quat_to_rad(a_head) + np.pi / 2

        angles = [0] * self.num_bins
        for person in self._sim.people:
            pos = person.current_position
            theta = math.atan2(pos[2] - a_pos[2], pos[0] - a_pos[0])
            theta = get_heading_error(a_head, theta)
            theta = theta if theta > 0 else 2 * np.pi + theta

            bin = int(theta / (2 * np.pi / self.num_bins))

            dist = np.sqrt((pos[2] - a_pos[2]) ** 2 + (pos[0] - a_pos[
                0]) ** 2)  # self._sim.geodesic_distance(a_pos, pos)
            norm_dist = max(1 - dist / self.thres, 0)
            if norm_dist > angles[bin]:
                angles[bin] = norm_dist

        return np.array(angles, dtype=np.float32)


@registry.register_task(name="SocialNav-v0")
class SocialNavigationTask(NavigationTask):
    def reset(self, episode: Episode):
        self._sim.reset_people()
        episode.people_paths = [
            p.waypoints
            for p in self._sim.people
        ]
        observations = super().reset(episode)

        if self._sim.sim_config.sim_cfg.enable_gfx_replay_save and "GFX_SAVE_DIR" in self._config:
            obj_templates_mgr = self._sim.get_object_template_manager()
            # get the rigid object manager, which provides direct
            # access to objects
            rigid_obj_mgr = self._sim.get_rigid_object_manager()

            # locobot
            locobot_template_id = obj_templates_mgr.load_configs(
                str("data/objects/locobot_merged")
            )[0]
            # add robot object to the scene with the agent/camera SceneNode attached
            locobot = rigid_obj_mgr.add_object_by_template_id(
                locobot_template_id, self._sim.agents[0].scene_node
            )
        return observations

    def step(self, action: Dict[str, Any], episode: Episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]

        # Move people
        for p in self._sim.people:
            p.step()

        #rig = self._sim.get_rigid_object_manager()
        #objects = [rig.get_object_by_id(rig.get_object_id_by_handle(h)) for h in rig.get_object_handles("person")]

        if self._sim.sim_config.sim_cfg.enable_gfx_replay_save and "GFX_SAVE_DIR" in self._config:
            gfx_replay.add_node_user_transform(self._sim, self._sim.get_agent(0).body.object, "agent")
            gfx_replay.add_node_user_transform(self._sim, self._sim._sensors["rgb"]._sensor_object.object, "sensor")
            self._sim.gfx_replay_manager.save_keyframe()


        observations = task_action.step(**action["action_args"], task=self)

        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        if not self._is_episode_active and "GFX_SAVE_DIR" in self._config:
            self._sim.gfx_replay_manager.write_saved_keyframes_to_file(self._config["GFX_SAVE_DIR"]+"/{}.{}.replay.json".format(self._sim.agents[0].initial_state.position, self._sim.config.sim_cfg.scene_id.split("/")[-1]))

        return observations


# # '_config', '_sim', '_dataset', 'measurements', 'sensor_suite',
# # 'actions', '_action_keys', 'is_stop_called'

#TODO: modify visualization of people's path (look into base classfRGB)
@registry.register_measure
class SocialTopDownMap(TopDownMap):

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "social_top_down_map"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        super().reset_metric(episode, *args, **kwargs)
        # Draw the paths of the people
        for person_path in episode.people_paths:
            map_corners = [
                maps.to_grid(
                    p[2],
                    p[0],
                    self._top_down_map.shape[0:2],
                    sim=self._sim,
                )
                for p in person_path
            ]
            maps.draw_path(
                self._top_down_map,
                map_corners,
                [255, 165, 0],  # Orange
                self.line_thickness,
            )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        people_map_coords = [
            maps.to_grid(
                p.current_position[2],
                p.current_position[0],
                self._top_down_map.shape[0:2],
                sim=self._sim,
            )
            for p in self._sim.people
        ]

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "people_map_coord": people_map_coords,
            "agent_angle": self.get_polar_angle(),
        }


@registry.register_measure
class HumanCollision(Measure):

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.uuid = "human_collision"

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "human_collision"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self._metric = False

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        agent_pos = self._sim.get_agent_state().position
        for p in self._sim.people:
            distance = np.sqrt(
                (p.current_position[0] - agent_pos[0]) ** 2
                + (p.current_position[2] - agent_pos[2]) ** 2
            )
            if distance < self._config.get('TERMINATION_RADIUS', 0.3):
                if "success" in task.measurements.measures and task.measurements.measures["success"].get_metric() == 1.0:
                    self._metric = False
                    task.is_stop_called = True
                    return
                else:
                    self._metric = True
                    break

        if self._metric:
            task.is_stop_called = True


@registry.register_measure
class PersonalSpaceCompliance(Measure):

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.uuid = "psc"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "psc"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self._metric = 0.0
        self._compliant_steps = 0
        self._num_steps = 0

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        #ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._num_steps += 1
        agent_pos = self._sim.get_agent_state().position
        compliance = True
        for p in self._sim.people:
            #self._sim.geodesic_distance()
            distance = self._sim.geodesic_distance(agent_pos, p.current_position)
            #distance = np.sqrt(
            #    (p.current_position[0] - agent_pos[0]) ** 2
            #    + (p.current_position[2] - agent_pos[2]) ** 2
            #)
            if distance < self._config.get('COMPLIANCE_RADIUS', 0.5):
                compliance = False
                break

        if compliance:
            self._compliant_steps += 1
        self._metric = (self._compliant_steps / self._num_steps)

@registry.register_measure
class PersonalSpaceComplianceH(Measure):

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.uuid = "psc_euc"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "psc_euc"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self._metric = 0.0
        self._compliant_steps = 0
        self._num_steps = 0

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any
    ):
        #ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._num_steps += 1
        agent_pos = self._sim.get_agent_state().position
        compliance = True
        for p in self._sim.people:
            #self._sim.geodesic_distance()

            distance = np.sqrt(
                (p.current_position[0] - agent_pos[0]) ** 2
                + (p.current_position[2] - agent_pos[2]) ** 2
            )
            if distance < self._config.get('COMPLIANCE_RADIUS', 0.5):
                compliance = False
                break

        if compliance:
            self._compliant_steps += 1
        self._metric = (self._compliant_steps / self._num_steps)
