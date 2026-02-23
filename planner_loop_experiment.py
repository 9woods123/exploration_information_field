import os
import numpy as np

from eif_map import *
from traj_generater import *

def sample_point(x_range, y_range):
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    return np.array([x, y])

def is_valid_configuration(start, mid, goal, min_dist=2.0):
    return (
        np.linalg.norm(start - mid)  > min_dist and
        np.linalg.norm(mid   - goal) > min_dist
    )

def run_experiment(seed, save_root="results_batch"):

    # ---------- reproducibility ----------
    np.random.seed(seed)

    # ---------- map & field generation ----------
    eif_table, sdf_field, map2d = map_generate(random_seed=22)

    # ---------- planner & optimizer ----------
    path_planner = PathPlanner(n_waypoints=30)

    lambda_info = 0.2
    lambda_col  = 0.5
    mu_smooth   = 1.0

    traj_optimizer = TrajOpti(
        eif_table, sdf_field,
        lambda_info, lambda_col, mu_smooth
    )

    traj_optimizer_sdfonly = TrajOpti(
        eif_table, sdf_field,
        0.0, lambda_col, mu_smooth
    )

    # START_X_RANGE = (-2.0, 2.0)
    # START_Y_RANGE = (-5.0, -8.0)

    # MID_X_RANGE   = ( -0.5,  0.5)
    # MID_Y_RANGE   = (-0.0,  2.0)

    # GOAL_X_RANGE  = (0.0, 7.0)
    # GOAL_Y_RANGE  = ( 5.0,  8.0)

    START_X_RANGE = (-5.0, 5.0)
    START_Y_RANGE = (-5.0, -6.0)

    MID_X_RANGE   = ( -0.5,  0.5)
    MID_Y_RANGE   = (-5,  5.0)

    GOAL_X_RANGE  = (-5, 5.0)
    GOAL_Y_RANGE  = ( 4,  7.0)
        # ---------- start / mid / goal ----------

    for _ in range(50):
        start = sample_point(START_X_RANGE, START_Y_RANGE)
        mid   = sample_point(MID_X_RANGE,   MID_Y_RANGE)
        goal  = sample_point(GOAL_X_RANGE,  GOAL_Y_RANGE)

        if is_valid_configuration(start, mid, goal):
            break


    traj0 = path_planner.init_polyline_traj(start, mid, goal)

    # ---------- optimization ----------
    traj_opt = traj_optimizer.optimize(
        traj0,
        n_iter=50,
        verbose=False
    )

    traj_sdfonly_opt = traj_optimizer_sdfonly.optimize(
        traj0,
        n_iter=50,
        verbose=False
    )

    # ---------- output path ----------
    os.makedirs(save_root, exist_ok=True)

    save_path = os.path.join(
        save_root, f"traj_eif_sdf_seed_{seed:04d}.pdf"
    )

    # ---------- visualization ----------
    plot_traj_and_fieldmap(
        eif_table,
        sdf_field,
        map2d,
        traj0,
        traj_opt,
        traj_sdfonly_opt,
        save_dir=save_root,
        frame=f"traj_eif_sdf_seed_{seed:04d}.pdf"
    )


def main():

    for seed in range(0, 1001):
        print(f"[INFO] Running experiment with seed = {seed}")
        run_experiment(seed)


if __name__ == "__main__":
    main()
