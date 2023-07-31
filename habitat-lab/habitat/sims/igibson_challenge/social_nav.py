# import collections
import os.path
from typing import Union, Optional, List

# from habitat.core.dataset import Dataset
# from habitat.core.embodied_task import Action, EmbodiedTask, Measure
# from habitat.core.simulator import ActionSpaceConfiguration, Sensor, Simulator
# from habitat.core.utils import Singleton

from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.config.default import Config

from habitat.utils.geometry_utils import (
    get_heading_error,
    quat_to_rad,
    heading_to_quaternion
)
from habitat.tasks.utils import cartesian_to_polar

import habitat_sim
from habitat_sim.geo import Ray
from habitat.sims.igibson_challenge.people_policy import PModel
import rvo2

from collections import deque

import random
import quaternion
import magnum as mn
import math
import numpy as np
import matplotlib.pyplot as plt
import torch as tc


@registry.register_simulator(name="iGibsonSocialNav")
class iGibsonSocialNav(HabitatSim):
    def __init__(self, config: Config) -> None:
        super().__init__(config=config)
        obj_templates_mgr = self.get_object_template_manager()
        self.people_template_ids = obj_templates_mgr.load_configs(
            "PATH_TO_PEOPLE_MESHES"
        )
        print(self.people_template_ids)
        self.person_ids = []
        self.people_mask = config.get('PEOPLE_MASK', False)
        self.num_people = config.get('NUM_PEOPLE', 1)  # can access navmesh and compute nav surface
        self.social_nav = True
        self.interactive_nav = False

        # People params
        self.people_mask = config.get('PEOPLE_MASK', False)
        self.lin_speed = config.PEOPLE_LIN_SPEED
        self.ang_speed = np.deg2rad(config.PEOPLE_ANG_SPEED)
        self.time_step = config.TIME_STEP

        # People policy
        self.people_policy = config.get('PPOLICY', "greedy")


    def reset_people(self):
        agent_position = self.get_agent_state().position
        obj_templates_mgr = self.get_object_template_manager()

        # Check if humans have been erased (sim was reset)
        if not self.get_existing_object_ids():
            self.person_ids = []
            for _ in range(self.num_people):
                for person_template_id in self.people_template_ids:
                    ind = self.add_object(person_template_id)
                    self.person_ids.append(ind)
                    self.set_object_semantic_id(len(self.person_ids)-1+600, ind)

        #print("people:", self.person_ids)
        #raise Exception("")
        # Spawn humans
        min_path_dist = 3
        max_level = 0.6
        agent_x, agent_y, agent_z = self.get_agent_state(0).position
        self.people = []
        for person_id in self.person_ids:
            valid_walk = False
            while not valid_walk:
                start = np.array(self.sample_navigable_point())
                goal = np.array(self.sample_navigable_point())
                distance = self.geodesic_distance(start, goal)
                valid_distance = distance > min_path_dist
                valid_level = (
                    abs(start[1]-agent_position[1]) < max_level
                    and abs(goal[1]-agent_position[1]) < max_level
                )
                sp = habitat_sim.nav.ShortestPath()
                sp.requested_start = start
                sp.requested_end   = goal
                found_path = self.pathfinder.find_path(sp)
                valid_start = np.sqrt(
                    (start[0]-agent_x)**2
                    +(start[2]-agent_z)**2
                ) > 0.5
                valid_walk = (
                    valid_distance and valid_level
                    and found_path and valid_start
                )
                if not valid_distance:
                    min_path_dist *= 0.95

            waypoints = sp.points
            heading = np.random.rand()*2*np.pi-np.pi
            rotation = np.quaternion(np.cos(heading),0,np.sin(heading),0)
            rotation = np.normalized(rotation)
            rotation = mn.Quaternion(
                rotation.imag, rotation.real
            )
            self.set_translation([start[0], start[1]+0.9, start[2]], person_id)
            self.set_rotation(rotation, person_id)
            self.set_object_motion_type(
                habitat_sim.physics.MotionType.KINEMATIC,
                person_id
            )
            if self.people_policy == "greedy":
                spf = ShortestPathFollowerv2(
                    sim=self,
                    object_id=person_id,
                    waypoints=waypoints,
                    lin_speed=self.lin_speed,
                    ang_speed=self.ang_speed,
                    time_step=self.time_step,
                )
            else:
                #future extension for RL trained people followers
                spf = DRLLongFollower(
                    sim=self,
                    object_id=person_id,
                    waypoints=waypoints,
                    lin_speed=self.lin_speed,
                    ang_speed=self.ang_speed,
                    time_step=self.time_step,
                )
            self.people.append(spf)

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:

        observations = super().get_observations_at(
            position,
            rotation,
            keep_agent_at_new_pose,
        )

        if observations is None:
            return None

        if not self.people_mask:
            return observations

        '''
        Get pixels of just people
        '''
        # 'Remove' people
        all_pos = []
        for person_id in self.get_existing_object_ids():
            pos = self.get_translation(person_id)
            all_pos.append(pos)
            self.set_translation([pos[0], pos[1]+10, pos[2]], person_id)

        # Refresh observations
        no_ppl_observations = super().get_observations_at(
            position=position,
            rotation=rotation,
            keep_agent_at_new_pose=True,
        )

        # Remove non-people pixels
        observations['people'] = observations['depth'].copy()
        observations['people'][
            observations['people'] == no_ppl_observations['depth']
        ] = 0

        # Put people back
        for pos, person_id in zip(all_pos, self.get_existing_object_ids()):
            self.set_translation(pos, person_id)

        return observations


class ShortestPathFollowerv2:
    def __init__(
        self,
        sim,
        object_id,
        waypoints,
        lin_speed,
        ang_speed,
        time_step,
    ):
        self._sim = sim
        self.object_id = object_id

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local    = True
        self.vel_control.ang_vel_is_local    = True

        self.waypoints = list(waypoints)+list(waypoints)[::-1][1:-1]
        self.next_waypoint_idx = 1
        self.done_turning = False
        self.current_position = waypoints[0]

        # People params
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.time_step = time_step
        self.max_linear_vel = np.random.rand()*(0.1)+self.lin_speed-0.1

    def step(self):
        waypoint_idx = self.next_waypoint_idx % len(self.waypoints)
        waypoint = np.array(self.waypoints[waypoint_idx])

        translation = self._sim.get_translation(self.object_id)
        mn_quat     = self._sim.get_rotation(self.object_id)

        # Face the next waypoint if we aren't already facing it
        if not self.done_turning:
            # Get current global heading
            heading = np.quaternion(mn_quat.scalar, *mn_quat.vector)
            heading = -quat_to_rad(heading)+np.pi/2

            # Get heading necessary to face next waypoint
            theta = math.atan2(
                waypoint[2]-translation[2], waypoint[0]-translation[0]
            )


            theta_diff = get_heading_error(heading, theta)
            direction = 1 if theta_diff < 0 else -1

            # If next turn would normally overshoot, turn just the right amount
            #print(abs(theta_diff))
            if self.ang_speed*self.time_step*1.2 >= abs(theta_diff):
                angular_velocity = -theta_diff / self.time_step
                self.done_turning = True
            else:
                angular_velocity = self.ang_speed*direction

            self.vel_control.linear_velocity = np.zeros(3)
            self.vel_control.angular_velocity = np.array([
                0.0, angular_velocity, 0.0
            ])

        # Move towards the next waypoint
        else:
            # If next move would normally overshoot, move just the right amount
            distance = np.sqrt(
                (translation[0]-waypoint[0])**2+(translation[2]-waypoint[2])**2
            )
            if self.max_linear_vel*self.time_step*1.2 >= distance:
                linear_velocity = distance / self.time_step
                self.done_turning = False
                self.next_waypoint_idx += 1
            else:
                linear_velocity = self.max_linear_vel

            #print("speed:", linear_velocity)
            self.vel_control.angular_velocity = np.zeros(3)
            self.vel_control.linear_velocity = np.array([
                0.0, 0.0, linear_velocity
            ])

        #print("angular:", self.vel_control.angular_velocity)
        #print("velocity:", self.vel_control.linear_velocity)
        rigid_state = habitat_sim.bindings.RigidState(
            mn_quat,
            translation
        )
        rigid_state = self.vel_control.integrate_transform(
            self.time_step, rigid_state
        )

        self._sim.set_translation(rigid_state.translation, self.object_id)
        self._sim.set_rotation(rigid_state.rotation, self.object_id)
        self.current_position = rigid_state.translation


class DRLLongFollower:
    def __init__(
        self,
        sim,
        object_id,
        waypoints,
        lin_speed,
        ang_speed,
        time_step,
    ):
        self._sim = sim
        self.object_id = object_id

        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        self.waypoints = list(waypoints) + list(waypoints)[::-1][1:-1]
        self.next_waypoint_idx = 1
        self.done_turning = False
        self.current_position = waypoints[0]

        waypoint = np.array(self.waypoints[1])
        theta = math.atan2(
            waypoint[2] - self.current_position[2], waypoint[0] - self.current_position[0]
        )
        n_rot = heading_to_quaternion(-theta)
        self._sim.set_rotation(n_rot, self.object_id)

        # People params
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.time_step = time_step
        self.max_linear_vel = np.random.rand() * (0.1) + self.lin_speed - 0.1

        self.rfinder_deque = None
        self.pmodel = PModel(frames=3, action_space=2).cuda()
        assert os.path.exists("PATH_TO_MODEL")
        state_dict = tc.load("PATH_TO_MODEL")
        self.pmodel.load_state_dict(state_dict, strict=False)
        self.speed = [0.0, 0.0]

    def get_range_finder_data(self, waypoint):
        mn_quat = self._sim.get_rotation(self.object_id)

        # Get current global heading
        heading = np.quaternion(mn_quat.scalar, *mn_quat.vector)
        heading = -quat_to_rad(heading) + np.pi / 2

        angles = np.linspace(heading - np.pi/2, heading + np.pi/2, 512)
        data = []
        points = []
        for a in angles:
            direction = mn.Vector3(np.cos(a), 0, np.sin(a))  #local coordinates wrt person
            ray = Ray(self.current_position, direction)
            r_res = self._sim.cast_ray(ray, max_distance=100.0)
            dist = min([h.ray_distance for h in r_res.hits], default=6.0)
            #point = min([(h.ray_distance, h.point) for h in r_res.hits], default=(0.0, 0.0), key=lambda k: k[1])[1]
            data.append(min(dist, 6.0)/6.0 - 0.5)
            #points.append(point)
            points.extend([h.point for h in r_res.hits])
        if self.object_id == 0:
            x = [p[0] for p in points]
            y = [p[2] for p in points]
            plt.scatter(x, y)
            plt.scatter([self._sim.get_translation(self.object_id)[0]], [self._sim.get_translation(self.object_id)[2]])
            plt.scatter([self._sim.get_translation(1)[0]],
                        [self._sim.get_translation(1)[2]], marker=".")
            plt.scatter([self._sim.get_translation(2)[0]],
                        [self._sim.get_translation(2)[2]], marker=".")
            plt.scatter([waypoint[0]], [waypoint[2]])
            plt.axline((self._sim.get_translation(self.object_id)[0], self._sim.get_translation(self.object_id)[2]), slope=np.tan(heading))
            plt.axline((self._sim.get_translation(self.object_id)[0],
                        self._sim.get_translation(self.object_id)[2]),
                       slope=np.tan(heading+np.pi/2))
            plt.axline((waypoint[0],
                        waypoint[2]),
                       slope=np.tan(heading + np.pi / 2))
            #plt.set_ylim(-6., +6.)
            plt.axis("equal")
            plt.show()
        return data

    def step(self):
        waypoint_idx = self.next_waypoint_idx % len(self.waypoints)
        waypoint = np.array(self.waypoints[waypoint_idx])

        translation = self._sim.get_translation(self.object_id)
        mn_quat = self._sim.get_rotation(self.object_id)

        distance = np.sqrt(
            (translation[0] - waypoint[0]) ** 2 + (
                translation[2] - waypoint[2]) ** 2
        )

        if distance < 0.1:
            self.next_waypoint_idx += 1
            waypoint_idx = self.next_waypoint_idx % len(self.waypoints)
            waypoint = np.array(self.waypoints[waypoint_idx])

        u = self.get_range_finder_data(waypoint)  # maybe create sensor to get rangefinder data from simultion, for debug
        if self.rfinder_deque is None:
            self.rfinder_deque = deque([u, u, u], maxlen=3)
        else:
            self.rfinder_deque.append(u)

        heading = np.quaternion(mn_quat.scalar, *mn_quat.vector)
        heading = -quat_to_rad(heading) + np.pi / 2

        #transform into local coordinates
        local_x = (waypoint[0] - translation[0]) * np.cos(heading) + (waypoint[2] - translation[2]) * np.sin(heading)
        local_y = -(waypoint[0] - translation[0]) * np.sin(heading) + (waypoint[2] - translation[2]) * np.cos(heading)

        with tc.no_grad():
            x = tc.unsqueeze(tc.from_numpy(np.asarray(self.rfinder_deque)).float(), 0).cuda()
            w = tc.unsqueeze(tc.tensor([local_x, local_y], dtype=tc.float, requires_grad=False), 0).cuda()
            s = tc.unsqueeze(tc.tensor(self.speed, dtype=tc.float, requires_grad=False), 0).cuda()

            _, action = self.pmodel.forward(x, w, s)
            action = tc.squeeze(action)

            linear_velocity = action[0].item()*self.max_linear_vel
            angular_velocity = action[1].item()*self.ang_speed

        self.vel_control.linear_velocity = np.array([
            0.0, 0.0, linear_velocity
        ])
        self.vel_control.angular_velocity = np.array([
            0.0, angular_velocity, 0.0
        ])

        rigid_state = habitat_sim.bindings.RigidState(
            mn_quat,
            translation
        )
        rigid_state = self.vel_control.integrate_transform(
            self.time_step, rigid_state
        )

        self._sim.set_translation(rigid_state.translation, self.object_id)
        self._sim.set_rotation(rigid_state.rotation, self.object_id)
        #assert rigid_state.translation != self.current_position
        self.current_position = rigid_state.translation
        self.speed = action.detach().cpu().numpy()
