import mujoco
import mujoco.viewer
import numpy as np
import time


def generate_many_random_configs(num_configs=100000000):
    """
    Generate XML with MANY random robot configurations (100+).
    This gives the appearance of infinite random morphologies.
    """

    xml = """<mujoco>
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81" timestep="0.005"/>
    
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
        
        <body name="base" pos="0 0 0.2">
            <geom type="box" size="0.08 0.08 0.05" rgba="0.5 0.5 0.5 1"/>
"""

    colors = ["0.2 0.4 0.8 1", "0.8 0.2 0.2 1", "0.2 0.8 0.2 1", "0.8 0.8 0.2 1"]
    joint_types = [2, 0, 2, 1]

    all_lengths = []

    print(f"Generating {num_configs} random robot configurations...")

    for config_id in range(num_configs):
        # Random lengths for THIS config
        lengths = np.random.uniform(0.15, 0.7, size=4)
        all_lengths.append(lengths)

        current_z = 0.05

        for link_id in range(4):
            if joint_types[link_id] == 0:
                axis = "1 0 0"
            elif joint_types[link_id] == 1:
                axis = "0 1 0"
            else:
                axis = "0 0 1"

            length = lengths[link_id]

            xml += f"""            <body name="c{config_id}_link{link_id}" pos="0 0 {current_z}">
                <joint name="c{config_id}_j{link_id}" type="hinge" axis="{axis}" range="-1.57 1.57" damping="0.5"/>
                <geom name="c{config_id}_g{link_id}" type="capsule" size="0.0001" fromto="0 0 0 0 0 {length}" 
                      rgba="{colors[link_id]}" mass="0.001" contype="0" conaffinity="0"/>
"""
            current_z = length

        xml += f"""                <site name="c{config_id}_ee" pos="0 0 {current_z}" size="0.0001" rgba="1 0 0 1"/>
"""

        for _ in range(4):
            xml += "            </body>\n"

        if (config_id + 1) % 20 == 0:
            print(f"  Generated {config_id + 1}/{num_configs}...")

    xml += """        </body>
    </worldbody>
    
    <actuator>
"""

    for config_id in range(num_configs):
        for link_id in range(4):
            xml += f'        <motor name="c{config_id}_m{link_id}" joint="c{config_id}_j{link_id}" ctrlrange="-3 3" gear="20"/>\n'

    xml += """    </actuator>
</mujoco>"""

    return xml, all_lengths


def main():
    print("=" * 70)
    print("DYNAMIC ROBOT MORPHOLOGY - 100 RANDOM CONFIGS")
    print("=" * 70)

    num_configs = 1000
    interval = 3.0

    xml, all_lengths = generate_many_random_configs(num_configs)

    print("\nBuilding MuJoCo model (this may take a moment)...")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print("✓ Model built successfully!")

    # Get IDs for each config
    configs = []
    for config_id in range(num_configs):
        config = {"geom_ids": [], "joint_ids": [], "motor_ids": [], "site_id": None}

        for link_id in range(4):
            config["geom_ids"].append(model.geom(f"c{config_id}_g{link_id}").id)
            config["joint_ids"].append(model.joint(f"c{config_id}_j{link_id}").id)
            config["motor_ids"].append(model.actuator(f"c{config_id}_m{link_id}").id)

        config["site_id"] = model.site(f"c{config_id}_ee").id
        configs.append(config)

    def hide_config(config_id):
        """Make config invisible"""
        cfg = configs[config_id]

        for geom_id in cfg["geom_ids"]:
            model.geom_size[geom_id][0] = 0.0001

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
            model.geom_size[geom_id][0] = 0.025

        model.site_size[cfg["site_id"]][0] = 0.03

        for joint_id in cfg["joint_ids"]:
            data.qpos[joint_id] = np.random.uniform(-0.8, 0.8)
            data.qvel[joint_id] = 0.0

    # Start with random config
    current_config = np.random.randint(0, num_configs)
    for i in range(num_configs):
        if i == current_config:
            show_config(i)
        else:
            hide_config(i)

    mujoco.mj_forward(model, data)

    lengths = all_lengths[current_config]
    print(f"Robot 1 (Config {current_config})")
    print(
        f"   Links: [{lengths[0]:.2f}, {lengths[1]:.2f}, {lengths[2]:.2f}, {lengths[3]:.2f}] = {np.sum(lengths):.2f}m"
    )

    print(" Starting visualization - viewer will stay open!")
    print("   Each spawn is a different random robot from 100 possibilities\n")

    viewer = mujoco.viewer.launch_passive(model, data)

    last_change = time.time()
    robot_count = 1
    step = 0
    used_configs = set([current_config])

    try:
        while viewer.is_running():
            current_time = time.time()
            step += 1

            time_left = interval - (current_time - last_change)
            if step % 100 == 0 and time_left > 0:
                print(f"   Next robot in {time_left:.1f}s...")

            # Switch to new RANDOM config
            if current_time - last_change >= interval:
                robot_count += 1
                old_config = current_config
                old_lengths = all_lengths[old_config]

                hide_config(old_config)

                # Pick a random config we haven't used recently
                available = list(set(range(num_configs)) - used_configs)
                if len(available) < 10:  # Reset if running low
                    used_configs = set([old_config])
                    available = list(set(range(num_configs)) - used_configs)

                current_config = np.random.choice(available)
                used_configs.add(current_config)

                show_config(current_config)

                new_lengths = all_lengths[current_config]

                mujoco.mj_forward(model, data)

                print(f"\n{'='*70}")
                print(
                    f"🤖 Robot {robot_count} (Config {current_config}) - VIEWER STAYS OPEN!"
                )
                print(
                    f"   OLD: [{old_lengths[0]:.2f}, {old_lengths[1]:.2f}, {old_lengths[2]:.2f}, {old_lengths[3]:.2f}] = {np.sum(old_lengths):.2f}m"
                )
                print(
                    f"   NEW: [{new_lengths[0]:.2f}, {new_lengths[1]:.2f}, {new_lengths[2]:.2f}, {new_lengths[3]:.2f}] = {np.sum(new_lengths):.2f}m"
                )
                print(f"   Change: {np.sum(new_lengths) - np.sum(old_lengths):+.2f}m")
                print(f"{'='*70}")

                last_change = current_time

            # Control only active config
            if step % 50 == 0:
                cfg = configs[current_config]
                for motor_id in cfg["motor_ids"]:
                    data.ctrl[motor_id] = np.random.uniform(-1, 1)

            # Zero all other configs
            for i in range(num_configs):
                if i != current_config:
                    for motor_id in configs[i]["motor_ids"]:
                        data.ctrl[motor_id] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        viewer.close()
        print("\nDone!")


if __name__ == "__main__":
    main()
