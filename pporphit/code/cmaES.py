import cma          # pip install cma   (only new dependency)
import numpy as np
from multiprocessing import Pool
from robotArmEnv import robotArmEnv   # or just import the class if it's all in one file

activeTask = "wallMount"
DIM = 1 + 7 * 2                       # 15-D action space

def evaluate_design(x: np.ndarray) -> float:
    """x is already in [0,1] (CMA keeps it there)"""
    env = robotArmEnv(taskName=activeTask)
    obs, _ = env.reset()
    action = np.clip(x, 0.0, 1.0)
    _, reward, _, _, _ = env.step(action)
    return -reward                     # CMA-ES *minimizes*

# ====================== PARALLEL VERSION (recommended) ======================
def parallel_evaluate(X: list[np.ndarray]) -> list[float]:
    with Pool(24) as p:                # same 24 workers you already use
        return p.map(evaluate_design, X)

if __name__ == "__main__":
    np.random.seed(42)
    x0 = np.random.uniform(0, 1, DIM)
    sigma0 = 0.35

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 32,          # 2–4× dimension works great
        'maxiter': 2000,
        'verb_disp': 1,
        'verb_log': 10,
        'tolx': 1e-4,
        'tolfun': 1e-6,
    })

    print("Starting CMA-ES optimization...")
    while not es.stop():
        X = es.ask()                                 # sample new designs
        f = parallel_evaluate(X)                     # 32 parallel evaluations
        es.tell(X, f)                                # update CMA model

        es.logger.add()                              # nice progress plots
        es.disp()                                    # print every 10 generations

    # ====================== RESULTS ======================
    best_x = es.result.xbest
    best_f = es.result.fbest

    print("\n=== BEST DESIGN FOUND ===")
    print("Action vector:", best_x)
    print("Reward:", -best_f)

    # Quick final evaluation with the best design
    env = robotArmEnv(taskName=activeTask)
    _, r, _, _, _ = env.step(np.clip(best_x, 0, 1))
    print(f"Final evaluated reward: {r}")