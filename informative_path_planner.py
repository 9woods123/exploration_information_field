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



def plot_eif_and_sdf_with_traj(eif_table, sdf_field, map2d, traj0, traj_opt):
    xs = eif_table.xs
    ys = eif_table.ys
    I_grid = eif_table.I
    SDF = sdf_field.sdf

    X, Y = np.meshgrid(xs, ys, indexing='ij')

    unknown_xy = np.array(
        [map2d.grid_to_world(idx) for idx in map2d.unknown]
    )

    traj0_pts = traj0.waypoints
    traj_opt_pts = traj_opt.waypoints

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

    # ---- init trajectory ----
    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='white',
        linewidth=2,
        label='init traj'
    )
    ax.scatter(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        c='white',
        s=25,
        edgecolors='black',
        linewidths=0.5,
        zorder=4,
        label='init waypoints'
    )

    # ---- optimized trajectory ----
    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='optimized traj'
    )
    ax.scatter(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        c='red',
        s=30,
        edgecolors='black',
        linewidths=0.5,
        zorder=5,
        label='optimized waypoints'
    )

    # start / goal
    ax.scatter(
        traj_opt_pts[0, 0],
        traj_opt_pts[0, 1],
        c='lime',
        s=80,
        zorder=6,
        label='start'
    )
    ax.scatter(
        traj_opt_pts[-1, 0],
        traj_opt_pts[-1, 1],
        c='red',
        s=80,
        zorder=6,
        label='goal'
    )

    ax.set_title("EIF Field + Trajectory")
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)

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

    # ---- init trajectory ----
    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='white',
        linewidth=2,
        label='init traj'
    )
    ax.scatter(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        c='white',
        s=25,
        edgecolors='black',
        linewidths=0.5,
        zorder=4,
        label='init waypoints'
    )

    # ---- optimized trajectory ----
    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='optimized traj'
    )
    ax.scatter(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        c='red',
        s=30,
        edgecolors='black',
        linewidths=0.5,
        zorder=5,
        label='optimized waypoints'
    )

    ax.scatter(
        traj_opt_pts[0, 0],
        traj_opt_pts[0, 1],
        c='lime',
        s=80,
        zorder=6,
        label='start'
    )
    ax.scatter(
        traj_opt_pts[-1, 0],
        traj_opt_pts[-1, 1],
        c='red',
        s=80,
        zorder=6,
        label='goal'
    )

    ax.set_title("SDF + Trajectory")
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()




def main():

    eif_table, sdf_field, map2d= map_generate()

    path_planner = PathPlanner(n_waypoints=40)
    traj_optimizer = TrajOpti(eif_table, sdf_field)

    start = np.array([-4.5, -5.0])
    mid  = np.array([ 3.0,  -4.0])
    goal  = np.array([ -2.0,  6.0])

    # traj0 = path_planner.init_straight_traj(start, goal)
    traj0 = path_planner.init_polyline_traj(start, mid , goal)




    traj_opt = traj_optimizer.optimize(
        traj0,
        n_iter=100,
        verbose=True
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
