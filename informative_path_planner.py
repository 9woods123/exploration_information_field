from eif_map import *
from traj_generater import *



def map_generate():
    timer = Timer()

    resolution = 0.2
    H_thresh = 0.65 * np.log(2)

    # -------------------------
    # Hyper-parameters
    # -------------------------
    N_INFO_PTS    = 100
    N_VIEWPOINTS = 100
    KDE_BANDWIDTH = 0.8
    GRAD_EPS      = 0.2
    GRID_STEP     = 0.3
    MAP_BOUND=10
    # -------------------------
    # Sensor & Map
    # -------------------------
    sensor = SensorModel(
        alpha=np.pi/2,
        kf=10.0,
        kr=4.0,
        dmax=3.0
    )

    map2d = Map2D(resolution)

    map2d.init_rectangle_known(
        center=(0.0, 0.0),
        width=12.0,
        height=15.0,
        bound=MAP_BOUND
    )
    map2d.add_random_rectangular_obstacles(
        n_obs=5
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
        pts, w = sampler.sample(t, N_INFO_PTS)
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
    grad_est = GradientEstimator(field, eps=GRAD_EPS)
    timer.lap("KDE field construction")

    # -------------------------
    # Build lookup table grid
    # -------------------------
    xs = np.arange(-MAP_BOUND, MAP_BOUND, GRID_STEP)
    ys = np.arange(-MAP_BOUND, MAP_BOUND, GRID_STEP)

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



    return eif_table, sdf_field, map2d



def main():

    eif_table, sdf_field, map2d= map_generate()

    path_planner = PathPlanner(n_waypoints=30)


    lambda_info=0.2
    lambda_col=0.5
    mu_smooth=1

    traj_optimizer = TrajOpti(eif_table, sdf_field,lambda_info,lambda_col,mu_smooth)
    traj_optimizer_sdfonly = TrajOpti(eif_table, sdf_field,0,lambda_col,mu_smooth)


    # start = np.array([-4.5, -5.0])
    # mid  = np.array([ 5.2,  1.0])
    # goal  = np.array([ -4.0,  6.0])
    start = np.array([-3, -6.5])
    mid  = np.array([ 3,  -0.5])
    goal  = np.array([ -3.0,  6.0])
    # traj0 = path_planner.init_straight_traj(start, goal)
    traj0 = path_planner.init_polyline_traj(start, mid , goal)




    traj_opt = traj_optimizer.optimize(
        traj0,
        n_iter=50,
        verbose=True
    )


    traj_sdfonly_opt = traj_optimizer_sdfonly.optimize(
        traj0,
        n_iter=50,
        verbose=True
    )

    
    plot_traj_and_fieldmap(
        eif_table,
        sdf_field,
        map2d,
        traj0,
        traj_opt,
        traj_sdfonly_opt
    )




if __name__ == "__main__":
    main()
