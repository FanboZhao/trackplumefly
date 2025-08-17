from flybody.tasks.vision_flight import VisionFlightImitationWBPG
import numpy as np
from typing import Optional, Sequence

from scipy.spatial.transform import Rotation

class PlumeTracking(VisionFlightImitationWBPG):
    def __init__(self,
                 goal_position: Optional[Sequence[float]] = None,
                 **kwargs):
        super().__init__(goal_position = goal_position,
                         **kwargs)
        self._camera_initialized = False
        self._pillars_initialized = False

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        
        mjcf_root = self.root_entity.mjcf_model

        if not self._pillars_initialized:
            self._num_pillars = 10
            for i in range(self._num_pillars):

                x = random_state.uniform(-6, 0)
                y = random_state.uniform(-2, 2)
            
                
                while abs(x + 4.5) < 1.0 and abs(y) < 1.0:
                    x = random_state.uniform(-5, 5)
                    y = random_state.uniform(-5, 5)
            
                
                height = random_state.uniform(0.5, 2.0)
                radius = random_state.uniform(0.05, 0.15)
            
                pillar_body = mjcf_root.worldbody.add(
                    'body', name=f'pillar_{i}',
                    pos=[x, y, height/2]
                )
            
                pillar_body.add(
                    'geom',
                    type='cylinder',
                    size=[radius, height/2],
                    rgba=[0.4, 0.2, 0.1, 1],
                    mass=10.0,
                    contype=1,
                    conaffinity=1
                )
            self._pillars_initialized = True

        # Initialize camera
        if not self._camera_initialized:
            mjcf_root.worldbody.add(
                'camera',
                name='tracking_cam',
                mode='fixed',
                pos=[0, -3, 2],
                xyaxes=[1, 0, 0, 0, 0, 1],
                fovy=60
            )
            self._camera_initialized = True
        
        return mjcf_root

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

    def before_step(self, physics, action, random_state):
    
        fly_pos, _ = self._walker.get_pose(physics)
    
        # calculate position of camera
        cam_pos = np.array(fly_pos) + np.array([2,0,0])
        # calculate direction of camera
        lookat = np.array([0,0,-1])
        
        cam_id = physics.model.name2id('tracking_cam', 'camera')
        physics.model.cam_pos[cam_id] = cam_pos
        physics.model.cam_quat[cam_id] = self._lookat_to_quat(lookat)
    
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        base_reward = super().get_reward_factors(physics)
    
        fly_pos, _ = self._walker.get_pose(physics)
    
        min_pillar_distance = float('inf')
        for i in range(self._num_pillars):
            pillar_pos = physics.named.data.xpos[f"pillar_{i}"]
            distance = np.linalg.norm(fly_pos - pillar_pos)
            if distance < min_pillar_distance:
                min_pillar_distance = distance
        
        safe_distance = 0.5 
        avoidance_penalty = 0.0
        if min_pillar_distance < safe_distance:
            avoidance_penalty = -0.3 * (safe_distance - min_pillar_distance)
    
    
        total_reward = base_reward + avoidance_penalty
    
        return total_reward

    def _lookat_to_quat(self, direction):
        """transform direction vector to quaternion"""
        direction = direction / np.linalg.norm(direction)
        default_dir = np.array([0, 0, -1])

        axis = np.cross(default_dir, direction)
        axis_norm = np.linalg.norm(axis)
        # If deviation is too small,return original quaternion
        if axis_norm < 1e-6:
            return np.array([1, 0, 0, 0])
        axis = axis / axis_norm
        angle = np.arccos(np.dot(default_dir, direction))

        return Rotation.from_rotvec(angle * axis).as_quat()