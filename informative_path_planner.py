from eif_map import *
from traj_generater import *



def map_generate():

    timer = Timer()

    resolution = 0.2
    H_thresh = 0.65 * np.log(2)

    # -------------------------
    # Hyper-parameters
    # -------------------------
    N_INFO_PTS    = 50
    N_VIEWPOINTS = 50
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
    Is = []
    for t in ts:
        pts, w = sampler.sample(t, N_INFO_PTS)
        _, I = evaluator.optimal_yaw_fast(t, pts, w)
        Is.append(I)

    Is = np.array(Is)
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


    # -------------------------
    # EIF lookup table
    # -------------------------
    eif_table = EIFLookupTable(xs, ys, I_grid, Gx_grid, Gy_grid)
    timer.lap("EIF table creation")



    # -------------------------
    # Validation query speed
    # -------------------------
    t_test = np.array([1.0, -1.0])

    N_QUERY = 5
    t0 = time.perf_counter()
    for _ in range(N_QUERY):
        eif_table.query_I(t_test)
        eif_table.query_grad(t_test)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    per_query_us = (t1 - t0) / N_QUERY * 1e6

    print(f"[QUERY] {N_QUERY} queries: {total_ms:.2f} ms "
        f"({per_query_us:.2f} µs / query)")



    return eif_table, sdf_field, map2d



def plot_eif_and_sdf_with_traj(eif_table, sdf_field, map2d, traj0, traj_opt):
    xs = eif_table.xs
    ys = eif_table.ys
    I_grid = eif_table.I
    SDF = sdf_field.sdf

    X, Y = np.meshgrid(xs, ys, indexing='ij')

    unknown_xy = np.array(
        [map2d.grid_to_world(idx) for idx in map2d.unknown]
    )

    traj0_pts = traj0.way_points
    traj_opt_pts = traj_opt.way_points

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # =====================================================
    # LEFT: EIF FIELD
    # =====================================================
    ax = axes[0]

    if len(unknown_xy) > 0:
        ax.scatter(
            unknown_xy[:, 0],
            unknown_xy[:, 1],
            s=5,
            c='lightgray',
            alpha=0.6,
            label='unknown'
        )

    c1 = ax.contourf(
        X, Y, I_grid,
        levels=30,
        cmap='viridis'
    )
    fig.colorbar(c1, ax=ax, shrink=0.8, label="EIF")

    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='white',
        linewidth=2,
        label='init traj'
    )

    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='optimized traj'
    )

    ax.scatter(
        traj_opt_pts[0, 0],
        traj_opt_pts[0, 1],
        c='lime',
        s=80,
        zorder=5
    )
    ax.scatter(
        traj_opt_pts[-1, 0],
        traj_opt_pts[-1, 1],
        c='red',
        s=80,
        zorder=5
    )

    ax.set_title("EIF Field + Trajectory")
    ax.set_aspect('equal')
    ax.legend()

    # =====================================================
    # RIGHT: SDF FIELD
    # =====================================================
    ax = axes[1]

    c2 = ax.contourf(
        X, Y, SDF,
        levels=40,
        cmap='coolwarm'
    )
    fig.colorbar(c2, ax=ax, shrink=0.8, label="SDF")

    # zero level set = obstacle boundary
    ax.contour(
        X, Y, SDF,
        levels=[0.0],
        colors='black',
        linewidths=2
    )

    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='white',
        linewidth=2,
        label='init traj'
    )

    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='optimized traj'
    )

    ax.scatter(
        traj_opt_pts[0, 0],
        traj_opt_pts[0, 1],
        c='lime',
        s=80,
        zorder=5
    )
    ax.scatter(
        traj_opt_pts[-1, 0],
        traj_opt_pts[-1, 1],
        c='red',
        s=80,
        zorder=5
    )

    ax.set_title("SDF + Trajectory")
    ax.set_aspect('equal')
    ax.legend()

    plt.tight_layout()
    plt.show()



def main():

    eif_table, sdf_field, map2d = map_generate()

    path_planner = PathPlanner()
    traj_optimizer = TrajOpti(eif_table, sdf_field)

    start = np.array([-4.0, -3.0])
    goal  = np.array([ 4.0,  3.0])

    traj0 = path_planner.init_straight_traj(start, goal)

    traj_opt = traj_optimizer.optimize(
        traj0,
        n_iter=60,
        verbose=False
    )

    # ========= NEW =========
    plot_eif_and_sdf_with_traj(
        eif_table,
        sdf_field,
        map2d,
        traj0,
        traj_opt
    )




if __name__ == "__main__":
    main()
