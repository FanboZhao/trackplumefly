from flybody.tasks.vision_flight import VisionFlightImitationWBPG
import numpy as np

from scipy.spatial.transform import Rotation

class BallVisionTrackingTask(VisionFlightImitationWBPG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ball_pos = np.zeros(3)
        self._ball_velocity = np.zeros(3)
        self._ball_init_pos = np.zeros(3)
        self._ball_initialized = False
        self._camera_initialized = False
        self._pillars_initialized = False

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        
        mjcf_root = self.root_entity.mjcf_model

        # put the ball in front of the fly       
        # Initialize tracked brown ball
        if not self._ball_initialized:
            ball_body = mjcf_root.worldbody.add(
                'body', name='target_ball',
                pos=[-4.5,0,0.7]
            )
        
            ball_body.add('joint', name='ball_joint', type='free')
            
            ball_body.add(
                'geom', 
                type='sphere', 
                size=[0.03],
                rgba=[1.0, 0.0, 0.0, 1.0], # red
                mass=0.05,
                friction=[1.0, 0.5, 0.01],
                contype=1,
                conaffinity=1,
                solref="0.02 1"
            )
            self._ball_initialized = True

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
        
        self._ball_pos = [-4.5,0,1]
        self._ball_vel = np.zeros(6)

    def before_step(self, physics, action, random_state):
        
        time = physics.data.time
        speed = 1
        amplitude = 5
        frequency = 0.5

        s_shape = amplitude * np.sin(frequency * time)
        
        target_pos = np.array([
            self._ball_init_pos[0] + speed * time,
            self._ball_init_pos[1] + s_shape,
            self._ball_init_pos[2],
        ])
    
        body_id = physics.model.name2id('target_ball', 'body')
        self._ball_pos = physics.data.xpos[body_id]
        self._ball_vel = physics.data.cvel[body_id]
    
        # parameters of PD controller.
        kp = np.array([15.0, 15.0, 30.0])
        kd = np.array([8.0, 8.0, 15.0])
    
        gravity_compensation = np.array([0, 0, 80])
        force = kp * (target_pos - self._ball_pos) - kd * self._ball_vel[:3] + gravity_compensation
    
        body_id = physics.model.name2id('target_ball', 'body')
        physics.data.xfrc_applied[body_id, :3] = np.clip(force, -100, 100)
    
        fly_pos, _ = self._walker.get_pose(physics)
    
        # calculate position of camera
        cam_pos = np.array((self._ball_pos + fly_pos) * 0.5)
        cam_pos[2] = 3
        # calculate direction of camera
        lookat = (self._ball_pos + fly_pos) * 0.5 - cam_pos
        
        cam_id = physics.model.name2id('tracking_cam', 'camera')
        physics.model.cam_pos[cam_id] = cam_pos
        physics.model.cam_quat[cam_id] = self._lookat_to_quat(lookat)
    
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        base_reward = super().get_reward_factors(physics)
    
        fly_pos, _ = self._walker.get_pose(physics)
        ball_distance = np.linalg.norm(self._ball_pos - fly_pos)
        tracking_reward = 0.6 * base_reward - 0.4 * ball_distance
    
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
    
    
        total_reward = tracking_reward + avoidance_penalty
    
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