# field_coverage.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加上级目录到路径
sys.path.append(os.path.abspath(".."))

from eif_map import *

mpl.rcParams['pdf.fonttype'] = 42     # TrueType
mpl.rcParams['ps.fonttype']  = 42


def generate_eif_map(map2d, viewpoint_num=50):
    timer = Timer()

    resolution = 0.2
    H_thresh = 0.65 * np.log(2)

    # -------------------------
    # Hyper-parameters
    # -------------------------
    N_INFO_PTS    = 100
    N_VIEWPOINTS = viewpoint_num
    KDE_BANDWIDTH = 0.8
    GRAD_EPS      = 0.2
    GRID_STEP     = 0.2
    MAP_BOUND=10
    N_SENSOR_RAYS=12
    # -------------------------
    # Sensor & Map
    # -------------------------
    sensor = SensorModel(
        alpha=np.pi/2,
        kf=10.0,
        kr=4.0,
        dmax=3.0
    )



    timer.lap("Map initialization")

    sampler   = InfoSampler(map2d, sensor, H_thresh)
    evaluator = EIFEvaluator(sensor)

    # -------------------------
    # Sample candidate viewpoints
    # -------------------------
    sel = np.random.choice(len(map2d.known), N_VIEWPOINTS, replace=False)
    ts = np.array([map2d.grid_to_world(map2d.known[i]) for i in sel])
    timer.lap("Viewpoint sampling")

    # -------------------------
    # EIF evaluation (MOST IMPORTANT)
    # -------------------------


    Yaw_grid= []
    Is = []
    for t in ts:
        pts, w = sampler.visibility_sample(t, N_SENSOR_RAYS)
        yaw_star, I_star = evaluator.optimal_yaw_fast(t, pts, w)
        Is.append(I_star)
        Yaw_grid.append(yaw_star)

    Is = np.array(Is)
    Yaw_grid= np.array(Yaw_grid)

    timer.lap("EIF evaluation @ viewpoints")

    # -------------------------
    # KDE continuous field
    # -------------------------
    field = KDEField(ts, Is, h=KDE_BANDWIDTH)
    # field = GroundTruthField(ts, Is)


    grad_est = GradientEstimator(field, eps=GRAD_EPS)
    timer.lap("KDE field construction")

    # -------------------------
    # Build lookup table grid
    # -------------------------
    xs = np.arange(-MAP_BOUND + 0.5*GRID_STEP, MAP_BOUND, GRID_STEP)
    ys = np.arange(-MAP_BOUND+  0.5*GRID_STEP, MAP_BOUND, GRID_STEP)

    nx, ny = len(xs), len(ys)
    I_grid  = np.full((nx, ny), np.nan)
    Gx_grid = np.zeros((nx, ny))
    Gy_grid = np.zeros((nx, ny))

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):

            t = np.array([x, y])

            # ---- ONLY free space ----
            if not map2d.is_free(t):
                I_grid[ix, iy]  = np.nan
                Gx_grid[ix, iy] = 0.0
                Gy_grid[ix, iy] = 0.0
                continue

            I_grid[ix, iy] = field.eval(t)

            g = grad_est.grad(t)
            Gx_grid[ix, iy] = g[0]
            Gy_grid[ix, iy] = g[1]

    timer.lap("Lookup table build (I + grad)")


    # -------------------------
    # Build SDF
    # -------------------------
    sdf_timer = Timer()

    sdf_field = SDF2D(
        map2d,
        xs,
        ys,
        resolution=resolution
    )

    sdf_field.build()
    sdf_timer.lap("SDF build")

    sdf_t_test = np.array([4, 5])
    print("sdf_field():",sdf_field.query(sdf_t_test))
    print("sdf_field()grad:",sdf_field.grad(sdf_t_test))

    # -------------------------
    # EIF lookup table
    # -------------------------


    Yaw_grid_2d = np.zeros((nx, ny))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):

            t = np.array([x, y])

            if not map2d.is_free(t):
                Yaw_grid_2d[ix, iy] = np.nan
                continue

            # -------- 找附近 viewpoints --------
            dists = np.linalg.norm(ts - t, axis=1)
            k = 5
            idx = np.argsort(dists)[:k]

            w = np.exp(-dists[idx]**2 / (2 * KDE_BANDWIDTH**2))
            yaws = Yaw_grid[idx]

            Yaw_grid_2d[ix, iy] = circular_mean(yaws, w)




    eif_table = EIFLookupTable(xs, ys, I_grid, Gx_grid, Gy_grid, Yaw_grid_2d)

    timer.lap("EIF table creation")



    return eif_table




def generate_baseline_map(map2d, viewpoint_num=50):


    timer = Timer()
    resolution=0.2
    H_thresh = 0.65 * np.log(2)

    # -------------------------
    # Hyper-parameters
    # -------------------------
    N_INFO_PTS    = 100
    N_VIEWPOINTS = 25
    KDE_BANDWIDTH = 0.8
    GRAD_EPS      = 0.2
    GRID_STEP     = 0.2
    MAP_BOUND=10
    N_SENSOR_RAYS=12
    # -------------------------
    # Sensor & Map
    # -------------------------
    sensor = SensorModel(
        alpha=np.pi/2,
        kf=10.0,
        kr=4.0,
        dmax=3.0
    )




    timer.lap("Map initialization")

    sampler   = InfoSampler(map2d, sensor, H_thresh)
    evaluator = EIFEvaluator(sensor)
    
    ts = [
        map2d.grid_to_world(idx)
        for idx in map2d.known
        if map2d.grid[idx] == 0.0   # free
    ]
    ts = np.array(ts)


    # -------------------------
    # EIF evaluation (MOST IMPORTANT)
    # -------------------------


    Yaw_grid= []
    Is = []
    for t in ts:
        pts, w = sampler.visibility_sample(t, N_SENSOR_RAYS)
        yaw_star_fast, I_star_fast = evaluator.optimal_yaw_fast(t, pts, w)
        yaw_star_gt, I_star_gt = evaluator.optimal_yaw_bruteforce(t, pts, w)

        Is.append(I_star_gt)
        Yaw_grid.append(yaw_star_gt)

    Is = np.array(Is)
    Yaw_grid= np.array(Yaw_grid)

    timer.lap("EIF evaluation @ viewpoints")

    # -------------------------
    # KDE continuous field
    # -------------------------
    field = KDEField(ts, Is,h= KDE_BANDWIDTH)
    
    grad_est = GradientEstimator(field, eps=GRAD_EPS)
    timer.lap("KDE field construction")

    # -------------------------
    # Build lookup table grid
    # -------------------------
    xs = np.arange(-MAP_BOUND + 0.5*GRID_STEP, MAP_BOUND, GRID_STEP)
    ys = np.arange(-MAP_BOUND+  0.5*GRID_STEP, MAP_BOUND, GRID_STEP)

    nx, ny = len(xs), len(ys)
    I_grid  = np.full((nx, ny), np.nan)
    Gx_grid = np.zeros((nx, ny))
    Gy_grid = np.zeros((nx, ny))

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):

            t = np.array([x, y])

            # ---- ONLY free space ----
            if not map2d.is_free(t):
                I_grid[ix, iy]  = np.nan
                Gx_grid[ix, iy] = 0.0
                Gy_grid[ix, iy] = 0.0
                continue

            I_grid[ix, iy] = field.eval(t)

            g = grad_est.grad(t)
            Gx_grid[ix, iy] = g[0]
            Gy_grid[ix, iy] = g[1]

    timer.lap("Lookup table build (I + grad)")




    # -------------------------
    # EIF lookup table
    # -------------------------



    Yaw_grid_2d = np.zeros((nx, ny))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):

            t = np.array([x, y])

            if not map2d.is_free(t):
                Yaw_grid_2d[ix, iy] = np.nan
                continue

            # -------- 找附近 viewpoints --------
            dists = np.linalg.norm(ts - t, axis=1)
            k = 5
            idx = np.argsort(dists)[:k]

            w = np.exp(-dists[idx]**2 / (2 * KDE_BANDWIDTH**2))
            yaws = Yaw_grid[idx]

            Yaw_grid_2d[ix, iy] = circular_mean(yaws, w)




    eif_table = EIFLookupTable(xs, ys, I_grid, Gx_grid, Gy_grid, Yaw_grid_2d)

    timer.lap("EIF table creation")



    return eif_table



# def compute_mse(eif1, eif2):
#     I1 = eif1.I
#     I2 = eif2.I

#     print("I1:", I1)
    
#     assert I1.shape == I2.shape, "Map size mismatch"

#     return np.mean((I1 - I2) ** 2)


def compute_mse(eif1, eif2):
    I1 = eif1.I
    I2 = eif2.I

    assert I1.shape == I2.shape, "Map size mismatch"

    # 只保留双方都不是 NaN 的区域
    valid_mask = (~np.isnan(I1)) & (~np.isnan(I2))

    if np.sum(valid_mask) == 0:
        print("No overlapping valid region!")
        return np.nan

    mse = np.mean((I1[valid_mask] - I2[valid_mask]) ** 2)

    print("valid pixels:", np.sum(valid_mask))
    print("total pixels:", I1.size)

    return mse

def compute_max_error(eif1, eif2):
    I1 = eif1.I
    I2 = eif2.I

    assert I1.shape == I2.shape, "Map size mismatch"

    valid_mask = (~np.isnan(I1)) & (~np.isnan(I2))

    if np.sum(valid_mask) == 0:
        return np.nan

    max_err = np.max(np.abs(I1[valid_mask] - I2[valid_mask]))

    return max_err

def run_experiment(
    scenario_name,
    map_builder,
    viewpoints_list,
    repeat=5
):
    print(f"\n===== Running scenario: {scenario_name} =====")

    resolution = 0.2
    scenario_map2d = Map2D(resolution)
    map_builder(scenario_map2d)

    print("Generating baseline map...")
    baseline_map = generate_baseline_map(map2d=scenario_map2d)

    # shape: [repeat, len(viewpoints)]
    all_max = []
    all_mse = []
    for r in range(repeat):

        mse_results = []
        max_results = []

        for n in viewpoints_list:

            eif_map = generate_eif_map(
                map2d=scenario_map2d,
                viewpoint_num=n,
            )

            mse = compute_mse(eif_map, baseline_map)
            max_err = compute_max_error(eif_map, baseline_map)

            mse_results.append(mse)
            max_results.append(max_err)

        all_mse.append(mse_results)
        all_max.append(max_results)

    all_mse = np.array(all_mse)

    all_mse = np.array(all_mse)
    all_max = np.array(all_max)

    mean_mse = np.nanmean(all_mse, axis=0)
    std_mse  = np.nanstd(all_mse, axis=0)

    mean_max = np.nanmean(all_max, axis=0)
    std_max  = np.nanstd(all_max, axis=0)

    return mean_mse, std_mse, mean_max, std_max


def build_t_corridor(map2d):
    map2d.init_T_corridor(
        center=(0.0, 0.0),
        w_vert=5.0,
        h_vert=18.0,
        w_horiz=14.0,
        h_horiz=4.0,
        wall_thickness=1,
        bound=12.0
    )


def build_dense_maze(map2d):
    map2d.init_dense_maze(
        K=4,
        cell_size=3.0,
        wall_thickness=0.5,
        seed=5
    )
    map2d.add_continuous_unknown(
        centers=[(0, 5), (4, -3.0)],
        radius=2
    )

def main():

    viewpoints_list = [
        10, 20, 30, 40, 50, 60,
        100, 110, 120, 130, 140,
        150, 160, 170, 180, 190, 200 ,250, 300, 350, 400
    ]

    # 场景 A
    mean_t, std_t, max_t ,std_max_t = run_experiment(
        scenario_name="T-Corridor",
        map_builder=build_t_corridor,
        viewpoints_list=viewpoints_list,
        repeat=3
    )

    mean_maze, std_maze, max_maze, std_max_maze= run_experiment(
        scenario_name="Maze",
        map_builder=build_dense_maze,
        viewpoints_list=viewpoints_list,
        repeat=3
    )

    plt.figure()

    # ---- T corridor ----
    plt.plot(viewpoints_list, mean_t, label="T-Corridor Mean")
    plt.fill_between(
    viewpoints_list,
    mean_t - std_t,
    mean_t + std_t,
    alpha=0.25
    )


    # ---- Maze ----
    plt.plot(viewpoints_list, mean_maze, label="Maze Mean")
    plt.fill_between(
    viewpoints_list,
    mean_maze - std_maze,
    mean_maze + std_maze,
    alpha=0.25
    )



    plt.xlabel("Number of Viewpoints")
    plt.ylabel("MSE to GT")
    plt.title("EIF Approximation Error vs Viewpoints")
    plt.legend()
    plt.grid(True)

    
    plt.show()



if __name__ == "__main__":
    main()