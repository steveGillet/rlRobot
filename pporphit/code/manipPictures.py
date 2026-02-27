import os
import glob
import numpy as np
import mujoco
import cv2
from stable_baselines3 import PPO
import math

# Force MuJoCo to use hardware-accelerated offscreen rendering on Linux
os.environ["MUJOCO_GL"] = "egl"

# --- Configuration (matching your robotArmEnv defaults) ---
MIN_LINKS = 2
MAX_LINKS = 7
MIN_LENGTH = 0.05
MAX_LENGTH = 1.2

def decode_action(action):
    """Decodes the PPO action vector back into morphology parameters."""
    numLinks = int(np.round(action[0] * (MAX_LINKS - MIN_LINKS) + MIN_LINKS))
    lengths = (action[1:(MAX_LINKS + 1)] * (MAX_LENGTH - MIN_LENGTH) + MIN_LENGTH)[:numLinks]
    jointTypes = np.round(action[(1 + MAX_LINKS):] * 3)[:numLinks].astype(int)
    return numLinks, lengths, jointTypes

def generate_neutral_xml(numJoints, lengths, jointTypes):
    """Generates a clean, neutral XML just for taking a picture."""
    
    total_length = sum(lengths)
    look_z = total_length * 0.45 
    cam_dist = max(1.2, total_length * 1.1) 
    
    # --- FLIPPED CAMERA ANGLE ---
    cam_x = -cam_dist * 0.707 
    cam_y = cam_dist * 0.707
    cam_z = look_z + (cam_dist * 0.4)

    # Note: Removed the floor, skybox, and grid textures entirely. 
    # Adjusted lighting to match the new camera position.
    xml = f"""
<mujoco>
    <compiler angle="radian" />
    <option gravity="0 0 -9.81" />
    <visual>
        <global offwidth="1920" offheight="1080" />
        <quality shadowsize="4096" numslices="32" numquads="4"/>
    </visual>
    <asset>
        <material name="robotMat" rgba="0.2 0.6 0.8 1" specular="0.5" shininess="0.5" />
    </asset>
    <worldbody>
        <light directional="true" pos="{cam_x} {cam_y} 3" dir="{-cam_x} {-cam_y} -2" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2" castshadow="true" />
        <light pos="0 0 2" dir="0 0 -1" diffuse="0.4 0.4 0.4" castshadow="false" />
        
        <body name="camera_target" pos="0 0 {look_z}"></body>
        <camera name="snapshot" mode="targetbody" target="camera_target" pos="{cam_x} {cam_y} {cam_z}" />

        <body name="base" pos="0 0 0.06">
            <geom name="baseBox" type="box" size="0.12 0.12 0.06" material="robotMat"/>
    """

    currentPos = "0 0 0.06"
    opened_bodies = 0 
    
    for i in range(numJoints):
        li = lengths[i]
        jCode = jointTypes[i]
        
        if jCode == 3:  # SLIDE JOINT
            axis, jtype = "0 0 1", "slide"
            xml += f"""
            <body name="link{i}_sleeve" pos="{currentPos}">
                <geom name="sleeve{i}" type="capsule" size="0.03" fromto="0 0 0 0 0 {li * 0.5:.4f}" material="robotMat" />
                <body name="link{i}_piston" pos="0 0 0">
                    <joint name="joint{i}" type="{jtype}" axis="{axis}" />
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {li:.4f}" material="robotMat" />
            """
            opened_bodies += 2
        else:  # HINGE JOINT
            if jCode == 0:
                axis, jtype = "1 0 0", "hinge"
            elif jCode == 1:
                axis, jtype = "0 1 0", "hinge"
            elif jCode == 2:
                axis, jtype = "0 0 1", "hinge"
            
            xml += f"""
            <body name="link{i}" pos="{currentPos}">
                <joint name="joint{i}" type="{jtype}" axis="{axis}" />
                <geom name="capsule{i}" type="capsule" size="0.03" fromto="0 0 0 0 0 {li:.4f}" material="robotMat" />
            """
            opened_bodies += 1
            
        currentPos = f"0 0 {li:.4f}"
        
    xml += "</body>\n" * opened_bodies
    xml += """
        </body>
    </worldbody>
</mujoco>
    """
    return xml

def main():
    output_dir = "rendered_morphologies"
    os.makedirs(output_dir, exist_ok=True)

    zip_files = glob.glob("*.zip")
    print(f"Found {len(zip_files)} zip files. Starting processing...")

    for zip_file in zip_files:
        model_name = os.path.splitext(zip_file)[0]
        output_path = os.path.join(output_dir, f"{model_name}.png")

        try:
            rl_model = PPO.load(zip_file, custom_objects={"lr_schedule": lambda x: .0, "clip_range": lambda x: .0})
            
            obs = np.array([0.0], dtype=np.float32)
            action, _ = rl_model.predict(obs, deterministic=True)
            
            numLinks, lengths, jointTypes = decode_action(action)
            print(f"[{model_name}] Decoded: {numLinks} links.")
            
            xml = generate_neutral_xml(numLinks, lengths, jointTypes)
            mj_model = mujoco.MjModel.from_xml_string(xml)
            mj_data = mujoco.MjData(mj_model)
            
            for i in range(numLinks):
                if jointTypes[i] == 3:
                    mj_data.qpos[i] = lengths[i] * 0.5
                else:
                    mj_data.qpos[i] = 0.6
                
            mujoco.mj_forward(mj_model, mj_data)
            mujoco.mj_step(mj_model, mj_data)
            
            renderer = mujoco.Renderer(mj_model, height=1080, width=1920)
            
            # 1. Render standard RGB image
            renderer.update_scene(mj_data, camera="snapshot")
            pixels_rgb = renderer.render()
            
            # 2. Render segmentation mask to find the background
            renderer.enable_segmentation_rendering()
            renderer.update_scene(mj_data, camera="snapshot")
            seg = renderer.render()
            
            # In MuJoCo, the background geom ID is -1.
            # seg is a (H, W, 2) array where the second channel is the geom ID.
            bg_mask = (seg[:, :, 1] == -1)
            
            # 3. Create a 4-channel BGRA image for OpenCV
            pixels_bgra = cv2.cvtColor(pixels_rgb, cv2.COLOR_RGB2BGRA)
            
            # 4. Set the alpha channel to 0 (transparent) wherever the background is
            pixels_bgra[bg_mask, 3] = 0
            
            cv2.imwrite(output_path, pixels_bgra)
            
            print(f"Saved: {output_path}")

            renderer.close()

        except Exception as e:
            print(f"Skipping {zip_file}: {e}")

    print(f"Done! Check the '{output_dir}' directory.")

if __name__ == "__main__":
    main()