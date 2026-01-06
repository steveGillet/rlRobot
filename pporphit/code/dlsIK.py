import numpy as np
import mujoco
import mujoco.viewer

def generateXML(numJoints, lengths, jointTypes):
    try:
        xml = """
<mujoco>
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="3 1 2" dir="0 0 -1" cutoff="180"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 0.5 0 1"/>
        <!-- <geom name="containerBack" type="box" pos="-2.0 0.6 1.0" size="0.01 1.0 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="containerLeft" type="box" pos="0 -0.4 1.0" size="2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="containerRight" type="box" pos="0 1.6 1.0" size=" 2.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/> -->
        <!-- <geom name="containerTop" type="box" pos="0 0.6 2.0" size="2.0 1.0 0.01" rgba="0.5 0.5 0.5 1"/> -->
        <geom name="mountWall" type="box" pos="0 -0.4 1.0" size="1.0 0.01 1.0" rgba="0.5 0.5 0.5 1"/>
        <geom name="shelfWall" type="box" pos="0 1.6 1.0" size=" 2.0 0.01 0.5" rgba="0.5 0.5 0.5 1"/>
        <geom name="shelf" type="box" pos="0.0 1.35 1.0" size="2.0 0.25 0.01" rgba="0.5 0.5 0.5 1"/>
        <body name="base" pos="0 -0.4 1.0" euler="-1.57 0 0">
        <!-- <geom name="obstacle" type="box" pos="0.45 0.25 0.55" size="0.3 0.1 0.025" rgba="1 0.5 0 1" /> -->
        <!-- <body name="base" pos="0 0 0"> -->
            <geom name="baseBox" type="box" size="0.1 0.1 0.05"/>
        """
        currentPos = "0 0 0.05"
        numCloses = 0
        for i in range(numJoints):
            if jointTypes[i] == 0:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="1 0 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses +=1
            elif jointTypes[i] == 1:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 1 0" range="-2.355 2.355" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            elif jointTypes[i] == 2:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <joint name="joint{i}" type="hinge" axis="0 0 1" damping="1.0"/>
                    <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]}"/>
                """
                currentPos = f"0 0 {lengths[i]}"
                numCloses += 1
            else:
                xml += f"""
                <body name="link{i}" pos="{currentPos}">
                    <geom name="baseCapsule{i}" type="capsule" size="0.025" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]/2}"/>
                    <body name="slideChild{i}"> 
                        <joint name="joint{i}" type="slide" axis="0 0 1" range="0 {lengths[i]}" damping="1.0"/>
                        <geom name="capsule{i}" type="capsule" size="0.02" fromto="0 0 0 0 0 {lengths[i]}" mass="{lengths[i]/2}"/>
                """
                currentPos = f"0 0 {lengths[i]}"    
                numCloses += 2
        xml += f'<site name="endEffector" pos="{currentPos}" size="0.01" rgba="0 1 0 1"/>'
        xml += "</body>" * numCloses  # Close links
        xml += """
        </body>  <!-- Close base -->
    <site name="startPos0" pos="0 1 -1" size="0.02" rgba="0 0 1 1"/>
    <site name="goalPos0" pos="-2 0 -1" size="0.02" rgba="1 0 0 1"/>
    <site name="startPos1" pos="0 1 -1" size="0.02" rgba="0 0 1 1"/>
    <site name="goalPos1" pos="-2 0 -1" size="0.02" rgba="1 0 0 1"/>    
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

numLinks = 7
lengths = np.array([0.05, 0.05, 0.05, 0.05, 0.46852955, 0.05, 1.1999999])
jointTypes = np.array([3, 0, 1, 2, 3, 3, 0])

# numLinks = 2
# lengths = np.array([0.46852955, 1.1999999])
# jointTypes = np.array([3, 0])

xml = generateXML(numLinks, lengths, jointTypes)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
obstacleIds = set([model.geom(name).id for name in ["mountWall", "shelfWall", "shelf", "floor"]])

targetPosition = np.array([-0.9, 1.65, 1.25], dtype=np.float32)
alpha = 0.75
maxIter = 200
lam = 0.01

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
endEffectorId = model.site("endEffector").id
mujoco.mj_forward(model, data)
deltaX = targetPosition - data.site(endEffectorId).xpos.copy()
i = 0

while i < maxIter and np.linalg.norm(deltaX) > 0.01:
    mujoco.mj_jacSite(model,data,jacp,jacr,endEffectorId)
    J = jacp

    U, Sigma, VT = np.linalg.svd(J, compute_uv=True, full_matrices=False)
    D = np.diag(Sigma / (Sigma**2 + lam**2))

    deltaTheta = VT.T @ D @ U.T @ deltaX

    collision = True
    al = alpha
    originalQpos = data.qpos.copy()

    while collision:
        collision = False

        data.qpos = originalQpos + al * deltaTheta
        data.qpos = np.clip(data.qpos, model.jnt_range[:, 0], model.jnt_range[:, 1])
        mujoco.mj_forward(model,data)
        
        for j in range(data.ncon):
            contact = data.contact[j]
            if contact.geom1 in obstacleIds or contact.geom2 in obstacleIds:
                collision = True
                print("collision")
                break

        al /= 2.0
        if al < 0.01:
            break

    if collision:
        data.qpos = originalQpos
        mujoco.mj_forward(model,data)
        print("Failed to get a collision free IK")
        break

    deltaX = targetPosition - data.site(endEffectorId).xpos.copy()
    i+=1


print(data.qpos)
print(data.site(endEffectorId).xpos.copy())

viewer = mujoco.viewer.launch_passive(model, data)

viewer.cam.lookat[:] = model.stat.center
viewer.cam.distance = model.stat.extent * 2
viewer.cam.elevation = -35
viewer.cam.azimuth = 145

viewer.sync()

input("Press enter to continue...")

print("Sim Complete")