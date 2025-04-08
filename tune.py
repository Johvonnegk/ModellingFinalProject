import pandas as pd
import numpy as np
from tqdm import tqdm
from simulate import simulate, gravity_options

def tune_k_for_config(mass, damping, gravity, angle_deg, k_range=np.linspace(5, 150, 30)):
    best_k = None
    best_time = float('inf')

    for k in k_range:
        t_settle = simulate(mass, damping, k, gravity, angle_deg)
        if t_settle is not None and t_settle <= 1.0 and t_settle < best_time:
            best_time = t_settle
            best_k = k

    return best_k, best_time if best_k else (None, None)


def run_simulation_with_tuning(target_successes=40, process_id=0):
    results = []
    success_count = 0
    trial = 0
    err = False
    pbar = tqdm(
        total=target_successes,
        desc=f"[Worker {process_id}]",
        position=process_id,
        leave=True,
        ncols=80
    )
    try:
        while success_count < target_successes:
            g = np.random.choice(gravity_options)
            m = np.random.uniform(0.5, 5.0)
            c = np.random.uniform(1.0, 20.0)
            angle = np.random.uniform(-15, 15)

            best_k, t_settle = tune_k_for_config(m, c, g, angle)

            success = best_k is not None and t_settle <= 1.0

            results.append({
                'mass': m,
                'damping': c,
                'gravity': g,
                'angle_deg': angle,
                'best_k': best_k if best_k else np.nan,
                'settling_time': t_settle if t_settle else np.nan,
                'success': success
            })

            if success:
                success_count += 1
                pbar.update()

            trial += 1

    except KeyboardInterrupt:
        err = True

    finally:
        pbar.close()
        df = pd.DataFrame(results)
        if not err:
            print(f"\n[Worker {process_id}] Collected {target_successes} successes in {trial} trials. Success rate: {target_successes / trial:.2%}")
        return df

