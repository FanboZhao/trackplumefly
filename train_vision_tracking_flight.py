from flybody.fly_envs import vision_tracking_flight, flight_imitation
from flybody.agents.agent_dmpo import DMPO
from flybody.agents.network_factory_vis import make_vis_network_factory_two_level_controller
from flybody.agents.network_factory import make_network_factory_dmpo
from flybody.agents.utils_tf import restore_dmpo_networks_from_checkpoint
from acme import specs, wrappers
from tqdm import tqdm
import mediapy

def wrap_env(env):
    """Wrap task environment with Acme wrappers."""
    return wrappers.CanonicalSpecWrapper(
        wrappers.SinglePrecisionWrapper(env),
        clip=True)

wpg_pattern_path = 'flybody-data/datasets_flight-imitation/wing_pattern_fmech.npy'
high_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/high-level-controllers/trench-task/ckpt-48'
low_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/low-level-controller/ckpt-11'

env = vision_tracking_flight(
        wpg_pattern_path=wpg_pattern_path,
        type='trench',
        joint_filter=0.0002,
    )

environment_spec = specs.make_environment_spec(env)

ll_env = wrap_env(flight_imitation(
    future_steps=5,
    joint_filter=0.0002))

ll_environment_spec = specs.make_environment_spec(ll_env)

ll_network_factory = make_network_factory_dmpo()


# Create networks for the vision flight task from their network factory.
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

agent = DMPO(environment_spec=environment_spec,
             policy_network=networks.policy_network,
             critic_network=networks.critic_network,
             observation_network=networks.observation_network,
            )

timestep = env.reset()
agent.observe_first(timestep)

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

timestep = env.reset()
frames = []
for _ in tqdm(range(10)):
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)
    frames.append(env.physics.render(camera_id=0, width=640, height=480))

mediapy.write_video('video/vision_tracking.mp4', frames, fps=30)
env.close()
