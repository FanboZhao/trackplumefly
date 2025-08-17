from flybody.fly_envs import flight_imitation
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
ref_path = 'flybody-data/datasets_flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5'

env = wrap_env(flight_imitation(
    ref_path=ref_path,
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

print(networks.observation_network)

agent = DMPO(environment_spec=environment_spec,
             policy_network=networks.policy_network,
             critic_network=networks.critic_network,
             observation_network=networks.observation_network,
            )
timestep = env.reset()

frames = []
for _ in tqdm(range(200)):
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)
    frames.append(env.physics.render(camera_id=3, width=640, height=480))

mediapy.write_video(f'video/simple_vision.mp4', frames, fps=30)

env.close()
