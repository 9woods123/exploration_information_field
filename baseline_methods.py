from eif_map import *




def generate_eifmap_GT():


    timer = Timer()

    resolution = 0.2
    H_thresh = 0.65 * np.log(2)

    # -------------------------
    # Hyper-parameters
    # -------------------------
    N_INFO_PTS    = 100
    N_VIEWPOINTS = 25
    KDE_BANDWIDTH = 0.8
    GRAD_EPS      = 0.2
    GRID_STEP     = 0.3
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


    map2d = Map2D(resolution)

    # map2d.init_T_corridor(
    #     center=(0.0, 0.0),
    #     w_vert=5.0,
    #     h_vert=18.0,
    #     w_horiz=14.0,
    #     h_horiz=4.0,
    #     wall_thickness=1,
    #     bound=12.0
    # )


    # map2d.init_rectangle_known(
    #     center=(0.0, 0.0),
    #     width=10.0,
    #     height=15.0,
    #     bound=MAP_BOUND
    # )

    # map2d.add_random_rectangular_obstacles(
    #     n_obs=8,
    #     w_range=(0.5, 4),
    #     h_range=(0.5, 4),
    #     seed=221221
    # )

    # map2d.init_T_corridor(
    #     center=(0.0, 0.0),
    #     w_vert=5.0,
    #     h_vert=18.0,
    #     w_horiz=14.0,
    #     h_horiz=4.0,
    #     wall_thickness=1,
    #     bound=12.0,
    #     wall_mode="both")


    map2d.init_dense_maze(K=4, cell_size=3.0, wall_thickness=0.5, seed=5)
    map2d.add_continuous_unknown(centers=[(0,5),(4,-3.0)], radius=2)



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
        yaw_star, I_star = evaluator.optimal_yaw_fast(t, pts, w)
        Is.append(I_star)
        Yaw_grid.append(yaw_star)

    Is = np.array(Is)
    Yaw_grid= np.array(Yaw_grid)

    timer.lap("EIF evaluation @ viewpoints")

    # -------------------------
    # KDE continuous field
    # -------------------------
    field = GroundTruthField(ts, Is)
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



    return eif_table, sdf_field, map2d



def main():

    eif_table, sdf_field, map2d= generate_eifmap_GT()
    plot_eif_and_sdf(eif_table, sdf_field,eif_table.Yaw,map2d ,show_sdf=False,
                     save_dir="results",
                     frame="eif_gt.pdf")



if __name__ == "__main__":
    main()