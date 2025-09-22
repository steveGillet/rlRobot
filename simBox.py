import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path('arm.xml')
data = mujoco.MjData(model)
shoulderMotorId = model.actuator('shoulderMotor').id
elbowMotorId = model.actuator('elbowMotor').id

simTime = 5.0
startTime = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < simTime:
        data.ctrl[shoulderMotorId] = -np.sin(data.time)*5
        data.ctrl[elbowMotorId] = np.cos(data.time)*3

        mujoco.mj_step(model,data)
        print(f"Time step: {data.time}s, Position: {data.geom_xpos[-1]}")
        viewer.sync()
        time.sleep(0.01)

print("Sim Complete")