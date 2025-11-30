import mujoco
import mujoco.viewer
import numpy as np
import time
import os

try:
    from stable_baselines3 import PPO

    SB3_AVAILABLE = True
except ImportError:
    print(
        "Warning: stable_baselines3 not found. Running in demo mode with random actions."
    )
    SB3_AVAILABLE = False

# Constants from robotArmEnv
MIN_NUM_LINKS = 2
MAX_NUM_LINKS = 7
MIN_LENGTH = 0.05
MAX_LENGTH = 1.5

# Set this to an integer (e.g. 4) to force a specific number of links,
# overriding the policy's choice. Set to None to use policy's choice.
FORCE_LINKS = 4


def decode_action(action, force_links=None):
    """
    Decodes the action from the PPO model into robot morphology parameters.
    Matches the logic in robotArmEnv.step()
    """
    if force_links is not None:
        numLinks = force_links
    else:
        numLinks = int(
            np.round(action[0] * (MAX_NUM_LINKS - MIN_NUM_LINKS) + MIN_NUM_LINKS)
        )
        # Clip numLinks to be safe
        numLinks = max(MIN_NUM_LINKS, min(MAX_NUM_LINKS, numLinks))

    lengths = (
        action[1 : (MAX_NUM_LINKS + 1)] * (MAX_LENGTH - MIN_LENGTH) + MIN_LENGTH
    )[:numLinks]
    jointTypes = np.round(action[(1 + MAX_NUM_LINKS) :] * 3)[:numLinks].astype(int)
    return numLinks, lengths, jointTypes


def generate_ppo_configs(model_path, num_configs=50):
    configs_data = []

    if SB3_AVAILABLE:
        print(f"Loading PPO model from {model_path}...")
        # Check if file exists
        if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
            print(f"Error: Model file {model_path} not found.")
            return None, None

        ppo_model = PPO.load(model_path)

        # Print PPO Model Statistics
        print("\n" + "=" * 70)
        print("PPO MODEL STATISTICS")
        print("=" * 70)
        print(f"Number of timesteps trained: {ppo_model.num_timesteps:,}")
        print(f"Learning rate: {ppo_model.learning_rate}")
        print(f"Gamma (discount factor): {ppo_model.gamma}")
        print(f"Batch size: {ppo_model.batch_size}")
        print(f"N steps: {ppo_model.n_steps}")
        print(f"N epochs: {ppo_model.n_epochs}")
        print(f"Clip range: {ppo_model.clip_range}")
        print(f"GAE lambda: {ppo_model.gae_lambda}")
        print(f"VF coefficient: {ppo_model.vf_coef}")
        print(f"Entropy coefficient: {ppo_model.ent_coef}")
        print(f"Max grad norm: {ppo_model.max_grad_norm}")

        # Policy network architecture
        print("\nPolicy Network Architecture:")
        print(f"  {ppo_model.policy}")

        # Action/Observation space
        print(f"\nAction Space: {ppo_model.action_space}")
        print(f"Observation Space: {ppo_model.observation_space}")
        print("=" * 70 + "\n")

        print(f"Generating {num_configs} configurations from PPO policy...")
    else:
        print(f"Generating {num_configs} random configurations (Demo Mode)...")
        ppo_model = None

    for i in range(num_configs):
        if ppo_model:
            # Observation is always [0.0] as per robotArmEnv.reset()
            obs = np.array([0.0], dtype=np.float32)
            # Sample from distribution to get variety
            action, _ = ppo_model.predict(obs, deterministic=False)

            # Get policy diagnostics (action distribution, value prediction)
            # Get the action distribution
            obs_tensor = ppo_model.policy.obs_to_tensor(obs)[0]
            ppo_model.policy.set_training_mode(False)
            distribution = ppo_model.policy.get_distribution(obs_tensor)
            # Get mean and std of the action distribution
            action_mean = (
                distribution.distribution.mean.detach().cpu().numpy().flatten()
            )
            action_std = (
                distribution.distribution.stddev.detach().cpu().numpy().flatten()
            )
            # Get value prediction
            values = ppo_model.policy.predict_values(obs_tensor)
            value_pred = values.detach().cpu().numpy().flatten()[0]
        else:
            # Random action
            # Shape: 1 + maxNumLinks * 2
            # 1 + 7*2 = 15
            action = np.random.uniform(0, 1, size=(1 + MAX_NUM_LINKS * 2))
            action_mean = action
            action_std = np.zeros_like(action)
            value_pred = 0.0

        numLinks, lengths, jointTypes = decode_action(action, force_links=FORCE_LINKS)

        # Capture raw actions for debugging
        raw_len_actions = (
            action[1 : (MAX_NUM_LINKS + 1)][:numLinks]
            if ppo_model
            else np.zeros(numLinks)
        )

        configs_data.append(
            {
                "id": i,
                "numLinks": numLinks,
                "lengths": lengths,
                "jointTypes": jointTypes,
                "raw_action_0": action[0] if ppo_model else 0,
                "raw_len_actions": raw_len_actions,
                "sampled_action": action,
                "action_mean": action_mean,
                "action_std": action_std,
                "value_pred": value_pred,
            }
        )

    # Generate XML
    # We use a structure similar to robotArmEnv but wrap multiple robots
    xml = """<mujoco>
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81" timestep="0.005"/>
    <size memory="100M"/>
    
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
        
        <!-- Obstacle from robotArmEnv -->
        <geom name="obstacle" type="box" pos="0.45 0.25 0.55" size="0.3 0.1 0.025" rgba="1 0.5 0 1" />
        
        <!-- Targets from robotArmEnv -->
        <site name="startPos" pos="0 1 -1" size="0.02" rgba="0 0 1 1"/>
        <site name="goalPos" pos="-2 0 -1" size="0.02" rgba="1 0 0 1"/>

        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.05" rgba="0.5 0.5 0.5 1"/>
"""

    colors = [
        "0.2 0.4 0.8 1",
        "0.8 0.2 0.2 1",
        "0.2 0.8 0.2 1",
        "0.8 0.8 0.2 1",
        "0.5 0.2 0.8 1",
        "0.2 0.8 0.8 1",
        "0.8 0.5 0.2 1",
    ]

    for cfg in configs_data:
        config_id = cfg["id"]
        numLinks = cfg["numLinks"]
        lengths = cfg["lengths"]
        jointTypes = cfg["jointTypes"]

        # Start height for first link (relative to base)
        # In robotArmEnv, base is at 0,0,0. First link body is at 0,0,0.05.
        current_z_start = 0.05

        xml += f"\n            <!-- Config {config_id} -->\n"

        # We construct the nested bodies
        # To do this linearly, we'll open tags, add content, and then close tags at the end.

        # We need to store the close tags to append later
        close_tags = ""

        for i in range(numLinks):
            length = lengths[i]
            j_type = jointTypes[i]

            if i == 0:
                pos = f"0 0 {current_z_start}"
            else:
                pos = f"0 0 {lengths[i-1]}"

            j_name = f"c{config_id}_j{i}"
            g_name = f"c{config_id}_g{i}"
            b_name = f"c{config_id}_link{i}"

            # Determine joint XML
            if j_type == 0:  # hinge 1 0 0
                joint_xml = f'<joint name="{j_name}" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="1.0"/>'
            elif j_type == 1:  # hinge 0 1 0
                joint_xml = f'<joint name="{j_name}" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>'
            elif j_type == 2:  # hinge 0 0 1
                joint_xml = (
                    f'<joint name="{j_name}" type="hinge" axis="0 0 1" damping="1.0"/>'
                )
            elif j_type == 3:  # slide 0 0 1
                joint_xml = f'<joint name="{j_name}" type="slide" axis="0 0 1" range="0 {length}" damping="1.0"/>'
            else:
                # Fallback
                joint_xml = (
                    f'<joint name="{j_name}" type="hinge" axis="0 0 1" damping="1.0"/>'
                )

            color = colors[i % len(colors)]

            # Added contype="0" conaffinity="0" to prevent collisions
            xml += f"""            <body name="{b_name}" pos="{pos}">
                {joint_xml}
                <geom name="{g_name}" type="capsule" size="0.02" fromto="0 0 0 0 0 {length}" rgba="{color}" mass="1.0" contype="0" conaffinity="0"/>
"""
            close_tags += "            </body>\n"

        # End effector site attached to the last link
        xml += f'                <site name="c{config_id}_ee" pos="0 0 {lengths[numLinks-1]}" size="0.01" rgba="0 1 0 1"/>\n'

        # Close all bodies
        xml += close_tags

    xml += """        </body>
    </worldbody>
    
    <actuator>
"""
    # Add actuators
    for cfg in configs_data:
        config_id = cfg["id"]
        numLinks = cfg["numLinks"]
        for i in range(numLinks):
            xml += f'        <motor name="c{config_id}_m{i}" joint="c{config_id}_j{i}" ctrlrange="-10 10"/>\n'

    xml += """    </actuator>
</mujoco>"""

    return xml, configs_data


def main():
    print("=" * 70)
    print("VISUALIZING PPO MORPHOLOGIES")
    print("=" * 70)

    model_path = "shelfArm"  # Assumes shelfArm.zip is in current dir
    num_configs = 50
    interval = 3.0

    xml, configs_data = generate_ppo_configs(model_path, num_configs)

    if xml is None:
        return

    print("\nBuilding MuJoCo model...")
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to build model: {e}")
        # Save XML for debugging
        with open("debug_ppo_morph.xml", "w") as f:
            f.write(xml)
        print("Saved debug_ppo_morph.xml")
        return

    print("✓ Model built successfully!")

    # Get IDs for each config to control visibility
    configs = []
    for cfg in configs_data:
        config_id = cfg["id"]
        numLinks = cfg["numLinks"]

        config_info = {
            "geom_ids": [],
            "joint_ids": [],
            "motor_ids": [],
            "site_id": None,
        }

        for i in range(numLinks):
            config_info["geom_ids"].append(model.geom(f"c{config_id}_g{i}").id)
            config_info["joint_ids"].append(model.joint(f"c{config_id}_j{i}").id)
            config_info["motor_ids"].append(model.actuator(f"c{config_id}_m{i}").id)

        config_info["site_id"] = model.site(f"c{config_id}_ee").id
        configs.append(config_info)

    def hide_config(config_id):
        """Make config invisible"""
        cfg = configs[config_id]

        for geom_id in cfg["geom_ids"]:
            model.geom_size[geom_id][0] = 0.0001  # Hide by making radius tiny

        model.site_size[cfg["site_id"]][0] = 0.0001

        for joint_id in cfg["joint_ids"]:
            data.qpos[joint_id] = 0.0
            data.qvel[joint_id] = 0.0

        for motor_id in cfg["motor_ids"]:
            data.ctrl[motor_id] = 0.0

    def show_config(config_id):
        """Make config visible"""
        cfg = configs[config_id]

        for geom_id in cfg["geom_ids"]:
            model.geom_size[geom_id][
                0
            ] = 0.02  # Restore radius (0.02 matches robotArmEnv)

        model.site_size[cfg["site_id"]][0] = 0.01  # Restore site size

        # Initialize with some random pose or zero
        for joint_id in cfg["joint_ids"]:
            # data.qpos[joint_id] = np.random.uniform(-0.5, 0.5)
            data.qpos[joint_id] = 0.0
            data.qvel[joint_id] = 0.0

    # Start with random config
    current_config = np.random.randint(0, num_configs)
    for i in range(num_configs):
        if i == current_config:
            show_config(i)
        else:
            hide_config(i)

    mujoco.mj_forward(model, data)

    print(f"Starting visualization. Switching every {interval} seconds.")

    viewer = mujoco.viewer.launch_passive(model, data)

    last_change = time.time()
    robot_count = 1
    step = 0
    used_configs = set([current_config])

    try:
        while viewer.is_running():
            current_time = time.time()
            step += 1

            # Switch to new RANDOM config
            if current_time - last_change >= interval:
                robot_count += 1
                old_config = current_config

                hide_config(old_config)

                # Pick a random config we haven't used recently
                available = list(set(range(num_configs)) - used_configs)
                if len(available) < 5:
                    used_configs = set([old_config])
                    available = list(set(range(num_configs)) - used_configs)

                current_config = np.random.choice(available)
                used_configs.add(current_config)

                show_config(current_config)

                # Print info
                cfg = configs_data[current_config]
                print(f"Robot {robot_count}: Config {current_config}")
                print(f"  Links: {cfg['numLinks']}")
                print(f"  Lengths: {np.round(cfg['lengths'], 2)}")
                print(f"  Types: {cfg['jointTypes']}")

                # PPO Policy Diagnostics
                if "value_pred" in cfg:
                    print(f"\n  --- PPO POLICY DIAGNOSTICS ---")
                    print(f"  Value Prediction: {cfg['value_pred']:.4f}")
                    print(
                        f"  Action Mean (policy center): {np.round(cfg['action_mean'][:4], 4)}"
                    )
                    print(
                        f"  Action Std (uncertainty):    {np.round(cfg['action_std'][:4], 4)}"
                    )
                    print(
                        f"  Sampled Action (actual):     {np.round(cfg['sampled_action'][:4], 4)}"
                    )

                    # Calculate action uncertainty
                    mean_uncertainty = np.mean(cfg["action_std"])
                    print(f"  Mean Action Uncertainty: {mean_uncertainty:.4f}")

                if "raw_action_0" in cfg:
                    print(f"\n  Raw Action[0]: {cfg['raw_action_0']:.4f}")
                if "raw_len_actions" in cfg:
                    print(
                        f"  Raw Length Actions: {np.round(cfg['raw_len_actions'], 4)}"
                    )
                print("-" * 30)

                mujoco.mj_forward(model, data)
                last_change = current_time

            # Random control for active config to show movement
            if step % 20 == 0:
                cfg = configs[current_config]
                for motor_id in cfg["motor_ids"]:
                    data.ctrl[motor_id] = np.random.uniform(-2, 2)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        viewer.close()
        print("Done!")


if __name__ == "__main__":
    main()
