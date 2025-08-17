import os
os.environ["MUJOCO_GL"] = "egl"

from flybody.fly_envs import simple_vision_guided_flight, flight_imitation
from flybody.agents.agent_dmpo import DMPO
from flybody.agents.network_factory_vis import make_vis_network_factory_two_level_controller
from flybody.agents.network_factory import make_network_factory_dmpo
from acme import specs, wrappers
from tqdm import tqdm
import mediapy
import sonnet as snt
import tensorflow as tf
from flybody.agents.utils_tf import restore_dmpo_networks_from_checkpoint

def wrap_env(env):
    """Wrap task environment with Acme wrappers."""
    return wrappers.CanonicalSpecWrapper(
        wrappers.SinglePrecisionWrapper(env),
        clip=True)

wpg_pattern_path = 'flybody-data/datasets_flight-imitation/wing_pattern_fmech.npy'
low_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/low-level-controller/ckpt-11'

env = wrap_env(flight_imitation(
    future_steps=5,
    joint_filter=0.0002))
environment_spec = specs.make_environment_spec(env)

# Create the same network architecture as in the flight_imitation task used to
# pre-train the low-level flight controller.
network_factory = make_network_factory_dmpo()

networks = restore_dmpo_networks_from_checkpoint(
    ckpt_path=low_level_ckpt_path,
    network_factory=network_factory,
    environment_spec=environment_spec)

class ObservationWrapper(snt.Module):
    def __init__(self, wrapped_network):
        super().__init__()
        self._network = wrapped_network

    def __call__(self, inputs):
        if isinstance(inputs, dict):
            inputs = {k: tf.cast(v, tf.float32) for k, v in inputs.items()}
        else:
            inputs = tf.cast(inputs, tf.float32)

        outputs = self._network(inputs)

        if isinstance(outputs, (tuple, list)):
            return [tf.cast(o, tf.float32) for o in outputs]
        return tf.cast(outputs, tf.float32)

networks.observation_network = ObservationWrapper(networks.observation_network)

agent = DMPO(environment_spec=environment_spec,
             policy_network=networks.policy_network,
             critic_network=networks.critic_network,
             observation_network=networks.observation_network,
            )
timestep = env.reset()
agent.observe_first(timestep)
'''
for i in range(10):
    timestep = env.reset()
    
    step = 0
    max_steps = 1000

    while step < max_steps:
        # print(f"Epoch {i + 1} Step {step} start")
        action = agent.select_action(timestep.observation)
        # print(f"Selected action: {action}")
        next_timestep = env.step(action)
        # print("Step completed, observing next timestep")
        agent.observe(action, next_timestep)
        # print(f"Reward: {next_timestep.reward}")
        agent.update()
        # print("Agent updated")
        if next_timestep.last():
            timestep = env.reset()
            # print("Environment reset")
            agent.observe_first(timestep)
            # print("Agent observed first timestep after reset")
        else:
            timestep = next_timestep
            # print(f"Step {step} completed")
        step += 1
    print(f"Step {(i + 1) * 1000} completed")

''' 
timestep = env.reset()

frames = []
for _ in tqdm(range(200)):
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)
    frames.append(env.physics.render(camera_id=3, width=640, height=480))

mediapy.write_video(f'video/simple_vision.mp4', frames, fps=30)

env.close()