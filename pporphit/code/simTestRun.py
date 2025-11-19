import simTest
env = simTest.ManipulatorEnv()
obs, _ = env.reset()
for _ in range(5):
    act = env.action_space.sample()
    obs, r, t, trunc, _ = env.step(act)
    print(r)