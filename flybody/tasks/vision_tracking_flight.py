from flybody.tasks.vision_flight import VisionFlightImitationWBPG

from flybody.fruitfly.fruitfly import FruitFly

import numpy as np
from acme import specs, wrappers

from typing import Callable, Sequence

import numpy as np

from dm_control import mujoco
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flybody.tasks.flight_imitation import FlightImitationWBPG
from flybody.tasks.vision_flight import VisionFlightImitationWBPG

from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.trajectory_loaders import (
    HDF5FlightTrajectoryLoader,
    InferenceFlightTrajectoryLoader,
)

def wrap_env(env):
    """Wrap task environment with Acme wrappers."""
    return wrappers.CanonicalSpecWrapper(
        wrappers.SinglePrecisionWrapper(env),
        clip=True)

def flight_imitation_walker(ref_path: str | None = None,
                     wpg_pattern_path: str | None = None,
                     force_actuators: bool = False,
                     disable_legs: bool = True,
                     traj_indices: Sequence[int] | None = None,
                     randomize_start_step: bool = True,
                     joint_filter: float = 0.,
                     future_steps: int = 5,
                     random_state: np.random.RandomState | None = None,
                     terminal_com_dist: float = 2.0):
    
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Initialize wing pattern generator and flight trajectory loader.
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    if ref_path is not None:
        traj_generator = HDF5FlightTrajectoryLoader(
            path=ref_path,
            traj_indices=traj_indices,
            randomize_start_step=randomize_start_step,
            random_state=random_state)
    else:
        traj_generator = InferenceFlightTrajectoryLoader()
    # Build the task.
    time_limit = 0.6
    task = FlightImitationWBPG(walker=walker,
                               arena=arena,
                               wbpg=wbpg,
                               traj_generator=traj_generator,
                               terminal_com_dist=terminal_com_dist,
                               initialize_qvel=True,
                               force_actuators=force_actuators,
                               disable_legs=disable_legs,
                               time_limit=time_limit,
                               joint_filter=joint_filter,
                               future_steps=future_steps)

    return task._walker

class VisionTrackingTask(VisionFlightImitationWBPG):
    def __init__(self, follower_wbpg, **kwargs):
        super().__init__(wbpg=follower_wbpg, **kwargs)

        from flybody.fly_envs import flight_imitation
        from flybody.agents.network_factory import make_network_factory_dmpo
        from flybody.agents.utils_tf import restore_dmpo_networks_from_checkpoint
        from flybody.agents.agent_dmpo import DMPO

        # Initialize virtual flight imitation environment
        low_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/low-level-controller/ckpt-11'
        ref_path = 'flybody-data/datasets_flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5'
        self._leader_env = wrap_env(flight_imitation(
            ref_path=ref_path,
            future_steps=5,
            joint_filter=0.0002))
        self._leader_timestep = None

        environment_spec = specs.make_environment_spec(self._leader_env)

        network_factory = make_network_factory_dmpo()

        networks = restore_dmpo_networks_from_checkpoint(
            ckpt_path=low_level_ckpt_path,
            network_factory=network_factory,
            environment_spec=environment_spec)

        # Initialize trained agent in flight imitation task
        self.leader_agent = DMPO(environment_spec=environment_spec,
                    policy_network=networks.policy_network,
                    critic_network=networks.critic_network,
                    observation_network=networks.observation_network,
                    )
        
        self.full_prev_action = None

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

        from flybody.tasks.flight_imitation import FlightImitationWBPG

        self._leader = flight_imitation_walker()
        mjcf_root = self.root_entity.mjcf_model
        # Initialize leader physics model
        # print("leader mjcf_model:", self._leader.mjcf_model)
        # 移除 leader 中的默认值块
        # leader_model = self._leader.mjcf_model
        # if leader_model.default:
            # del leader_model.default  # 或 selective del 里面的 motor/actuator 等
        
        self.root_entity.attach(self._leader)

        # 增加 Arena 内存
        # mjcf_root.option.set_attributes(nconmax="10000", njmax="1000")

  # 200 MB，原本可能只有几 MB

        # mjcf_root.attach(leader_model)
        return mjcf_root

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self.full_prev_action = physics.data.ctrl
        print(self.full_prev_action.shape)
        
        init_x = -4.5
        init_y = random_state.uniform(*self._init_pos_y_range)

        # Reset wing pattern generator and get initial wing angles.
        initial_phase = random_state.uniform()
        init_wing_qpos = self._wbpg.reset(initial_phase=initial_phase)

        self._arena.initialize_episode(physics, random_state)
        from flybody.tasks.task_utils import neg_quat
        # Initialize root position and orientation.
        hfield_height = self.get_hfield_height(init_x, init_y, physics)
        init_z = hfield_height + self._target_height
        self._leader.set_pose(physics, np.array([init_x, init_y, init_z]),
                              neg_quat(self._up_dir)) 

        # reset 虚拟 leader 环境
        self._leader_timestep = self._leader_env.reset()

    def before_step(self, physics, action, random_state):
        leader_action = self.leader_agent.select_action(self._leader_timestep.observation)
        
        self._leader_timestep = self._leader_env.step(leader_action)

        self.full_prev_action[14:26] = leader_action

        leader_action = self.full_prev_action

        # 传入完整59维动作
        self._leader.apply_action(physics, leader_action, random_state)

        super().before_step(physics, action, random_state)


