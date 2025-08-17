from flybody.fly_envs import ball_vision_tracking_flight, flight_imitation
from flybody.agents.agent_dmpo import DMPO
from flybody.agents.network_factory_vis import make_vis_network_factory_two_level_controller
from flybody.agents.network_factory import make_network_factory_dmpo
from flybody.agents.utils_tf import restore_dmpo_networks_from_checkpoint
from acme import specs, wrappers
from tqdm import tqdm
import mediapy
from dm_control import viewer
import os
import tensorflow as tf

def wrap_env(env):
    """Wrap task environment with Acme wrappers."""
    return wrappers.CanonicalSpecWrapper(
        wrappers.SinglePrecisionWrapper(env),
        clip=True)

checkpoint_dir = './saved_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

wpg_pattern_path = 'flybody-data/datasets_flight-imitation/wing_pattern_fmech.npy'
high_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/high-level-controllers/trench-task/ckpt-48'
low_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/low-level-controller/ckpt-11'

env = ball_vision_tracking_flight(
        wpg_pattern_path=wpg_pattern_path,
        bumps_or_trench='bumps',
        joint_filter=0.0002,
    )

environment_spec = specs.make_environment_spec(env)

ll_env = wrap_env(flight_imitation(
    future_steps=5,
    joint_filter=0.0002))

ll_environment_spec = specs.make_environment_spec(ll_env)

ll_network_factory = make_network_factory_dmpo()


"""Create networks for the vision flight task from their network factory."""
network_factory = make_vis_network_factory_two_level_controller(
    ll_network_ckpt_path=low_level_ckpt_path,
    ll_network_factory=ll_network_factory,
    ll_environment_spec=ll_environment_spec,
    hl_network_layer_sizes=(256, 256, 128),
    steering_command_dim=(5 + 1) * (3 + 4),
    task_input_dim=2,
    vis_output_dim=8,
    critic_layer_sizes=(512, 512, 256),
)

networks = restore_dmpo_networks_from_checkpoint(
    ckpt_path=high_level_ckpt_path,
    network_factory=network_factory,
    environment_spec=environment_spec)

"""Load the trained agent"""
agent = DMPO(environment_spec=environment_spec,
             policy_network=networks.policy_network,
             critic_network=networks.critic_network,
             observation_network=networks.observation_network,
            )

checkpoint = tf.train.Checkpoint(
    policy_network=networks.policy_network,
    critic_network=networks.critic_network,
    observation_network=networks.observation_network,
)
manager = tf.train.CheckpointManager(
    checkpoint, directory=checkpoint_dir, max_to_keep=3
)

print("Begin training!")
timestep = env.reset()
agent.observe_first(timestep)

save_every_n_episodes = 10

for i in range(2000):
    timestep = env.reset()
    step = 0
    max_steps = 1000

    while step < max_steps:
        action = agent.select_action(timestep.observation)
        next_timestep = env.step(action)
        agent.observe(action, next_timestep)
        agent.update()
        if next_timestep.last():
            timestep = env.reset()
            agent.observe_first(timestep)
        else:
            timestep = next_timestep
        step += 1
    print(f"Step {(i + 1) * 1000} completed!")

    if (i + 1) % save_every_n_episodes == 0:
        save_path = manager.save(checkpoint_number=i+1)
        print(f"Saved checkpoint for episode {i+1} at {save_path}")

print("Train accomplished!")
final_save_path = manager.save(checkpoint_number=500)
print(f"Saved final checkpoint at {final_save_path}")

# timestep = env.reset()
# viewer.launch(env, policy=lambda ts: agent.select_action(ts.observation))

"""Save the video."""
timestep = env.reset()
frames1 = []
frames2 = []

for _ in range(1000):
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)
    frames1.append(env.physics.render(camera_id='tracking_cam',width=640,height=480,))
    frames2.append(env.physics.render(camera_id=2, width=640, height=480))

mediapy.write_video('tracking_view01.mp4', frames1, fps=30)
mediapy.write_video('tracking_view02.mp4', frames2, fps=30)
env.close()
print("Finished!")