def generateContainerXML(numJoints, lengths, jointTypes):
    try:
        xml = """
<mujoco>
    <compiler angle="radian" />
    <option gravity="0 0 -9.81" />
    <asset>
        <!-- Builtin grid texture for floor -->
        <texture name="grid" type="2d" builtin="checker" rgb1="0.4 0.4 0.4" rgb2="0.6 0.6 0.6" width="512" height="512" />
        <material name="floorMat" texture="grid" texrepeat="10 10" specular="0.2" shininess="0.1" />
        <!-- Metallic material for robot -->
        <material name="robotMat" rgba="0.75 0.78 0.82 1" specular="0.8" shininess="0.9" />
        <!-- Forest green for container -->
        <material name="containerMat" rgba="0.15 0.35 0.15 0.88" specular="0.4" shininess="0.3" />
    </asset>
    <worldbody>
        <!-- Big sun light from above (bright, universal) -->
        <light name="sun" pos="3 2 9" dir="0 0 -1" diffuse="1.1 1.1 1.1" specular="0.8 0.8 0.8" cutoff="170" />
        <!-- Internal container light (soft, diffused) -->
        <light name="internal" pos="0 0.6 1.2" diffuse="0.55 0.65 0.45" specular="0.15 0.15 0.15" cutoff="140" />
        <!-- Floor with grid texture -->
        <geom name="floor" type="plane" size="3 3 0.1" material="floorMat" />
        <!-- Container obstacles (forest green) -->
        <geom name="containerBack" type="box" pos="-2.0 0.6 1.0" size="0.01 1.0 1.0" material="containerMat" />
        <geom name="containerLeft" type="box" pos="0 -0.4 1.0" size="2.0 0.01 1.0" material="containerMat" />
        <geom name="containerRight" type="box" pos="0 1.6 1.0" size="2.0 0.01 1.0" material="containerMat" />
        <geom name="containerTop" type="box" pos="0 0.6 2.0" size="2.0 1.0 0.01" material="containerMat" />
        <!-- Robot base -->
        <body name="base" pos="0 0 0.05">
            <geom name="baseBox" type="box" size="0.12 0.12 0.06" rgba="0.2 0.2 0.2 1" material="robotMat" />
        """
        currentPos = "0 0 0.05"
        numCloses = 0
        for i in range(numJoints):
            if jointTypes[i] == 0:  # X hinge
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="1 0 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 1:  # Y hinge
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 2:  # Z hinge
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 0 1" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            else:  # Slide Z
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <geom name="baseCapsule{i}" type="capsule" size="0.028" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                    <body name="slideChild{i}">
                        <joint name="joint{i}" type="slide" axis="0 0 1" range="0 {lengths[i]}" damping="1.0"/>
                        <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 2
        xml += f'<site name="endEffector" pos="{currentPos}" size="0.015" rgba="0 1 0 1" />'
        xml += "</body>" * numCloses
        xml += """
        </body>  <!-- Close base -->
        <!-- Start/Goal sites -->
        <site name="startPos0" pos="0 1 -1" size="0.025" rgba="0.2 0.8 1 1" />
        <site name="goalPos0" pos="-2 0 -1" size="0.025" rgba="1 0.3 0.2 1" />
        <site name="startPos1" pos="0 1 -1" size="0.025" rgba="0.2 0.8 1 1" />
        <site name="goalPos1" pos="-2 0 -1" size="0.025" rgba="1 0.3 0.2 1" />
    </worldbody>
    <actuator>
        """
        for i in range(numJoints):
            xml += f'<motor name="motor{i}" joint="joint{i}" ctrlrange="-10 10"/>'
        xml += """
    </actuator>
</mujoco>
        """
        return xml
    except Exception as e:
        print(f"Mujoco XML Generation Error: {e}")
        raise

def generateWallXML(numJoints, lengths, jointTypes):
    try:
        xml = """
<mujoco>
    <compiler angle="radian" />
    <option gravity="0 0 -9.81" />
    <asset>
        <!-- Builtin grid texture for floor -->
        <texture name="grid" type="2d" builtin="checker" rgb1="0.4 0.4 0.4" rgb2="0.6 0.6 0.6" width="512" height="512" />
        <material name="floorMat" texture="grid" texrepeat="10 10" specular="0.2" shininess="0.1" />
        <!-- Metallic material for robot -->
        <material name="robotMat" rgba="0.75 0.78 0.82 1" specular="0.8" shininess="0.9" />
        <!-- Forest green for container -->
        <material name="containerMat" rgba="0.75 0.58 0.42 0.88" specular="0.4" shininess="0.3" />
    </asset>
    <worldbody>
        <!-- Big sun light from above (bright, universal) -->
        <light name="sun" pos="-1 -1 9" dir="0 0 -1" diffuse="1.1 1.1 1.1" specular="0.8 0.8 0.8" cutoff="170" />
        <!-- Internal container light (soft, diffused) -->
        <light name="internal" pos="0 0.6 1.2" diffuse="0.55 0.65 0.45" specular="0.15 0.15 0.15" cutoff="140" />
        <!-- Floor with grid texture -->
        <geom name="floor" type="plane" size="3 3 0.1" material="floorMat" />
        <!-- Wall mount task obstacles (forest green) -->
        <geom name="mountWall" type="box" pos="0 -0.4 1.0" size="1.0 0.01 1.0" material="containerMat" />
        <geom name="shelfWall" type="box" pos="0 1.6 1.0" size="2.0 0.01 1.0" material="containerMat" />
        <geom name="shelf" type="box" pos="0.0 1.35 1.0" size="2.0 0.25 0.01" material="containerMat" />
        <!-- Robot base (rotated for wall mount) -->
        <body name="base" pos="0 -0.4 1.0" euler="-1.57 0 0">
            <geom name="baseBox" type="box" size="0.12 0.12 0.06" rgba="0.2 0.2 0.2 1" material="robotMat" />
        """
        currentPos = "0 0 0.05"
        numCloses = 0
        for i in range(numJoints):
            if jointTypes[i] == 0:  # X hinge
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="1 0 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 1:  # Y hinge
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 2:  # Z hinge
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 0 1" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            else:  # Slide Z
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <geom name="baseCapsule{i}" type="capsule" size="0.028" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                    <body name="slideChild{i}"> 
                        <joint name="joint{i}" type="slide" axis="0 0 1" range="0 {lengths[i]}" damping="1.0"/>
                        <geom name="capsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" material="robotMat" />
                """
                currentPos = f"0 0 {lengths[i]}"    
                numCloses += 2
        xml += f'<site name="endEffector" pos="{currentPos}" size="0.015" rgba="0 1 0 1" />'
        xml += "</body>" * numCloses
        xml += """
        </body>  <!-- Close base -->
        <!-- Start/Goal sites -->
        <site name="startPos0" pos="-0.9 1.35 1.1" size="0.025" rgba="0.2 0.8 1 1" />
        <site name="goalPos0" pos="1.5 -0.4 0.2" size="0.025" rgba="1 0.3 0.2 1" />
        <site name="startPos1" pos="-1.5 -0.4 0.1" size="0.025" rgba="0.2 0.8 1 1" />
        <site name="goalPos1" pos="1.75 1.36 1.11" size="0.025" rgba="1 0.3 0.2 1" />
    </worldbody>
    <actuator>
        """
        for i in range(numJoints):
            xml += f'<motor name="motor{i}" joint="joint{i}" ctrlrange="-10 10"/>'
        xml += """
    </actuator>
</mujoco>
        """
        return xml
    except Exception as e:
        print(f"Mujoco XML Generation Error: {e}")
        raise
