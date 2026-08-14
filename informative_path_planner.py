from eif_map import *
from traj_generater import *






def main():
### TTTTTTTTTTTTTTTT
## FOLLOWING PARAMETERS ARE SET FOR T CORIDOR EXPERIMENT
# DO NOT CHANGE

    eif_table, sdf_field, map2d= map_generate(random_seed=6)

    path_planner = PathPlanner(n_waypoints=40)


    lambda_info=0.25
    lambda_col=0.5
    mu_smooth=1

    traj_optimizer = TrajOpti(eif_table, sdf_field,lambda_info,lambda_col,mu_smooth)
    traj_optimizer_sdfonly = TrajOpti(eif_table, sdf_field,0,lambda_col,mu_smooth)


    # start = np.array([-4.5, -5.0])
    # mid  = np.array([ 5.2,  1.0])
    # goal  = np.array([ -4.0,  6.0])
    # start = np.array([-0, -6.5])
    # mid  = np.array([ 1.5,  0.55])
    # goal  = np.array([ -3.0,  5.0])



    start = np.array([1, -8.5])
    mid  = np.array([ 0.35, 3])
    goal  = np.array([ -6.0,  7.5])
    
    # traj0 = path_planner.init_straight_traj(start, goal)
    traj0 = path_planner.init_polyline_traj(start, mid , goal)




    traj_opt = traj_optimizer.optimize(
        traj0,
        n_iter=100,
        verbose=True
    )


    traj_sdfonly_opt = traj_optimizer_sdfonly.optimize(
        traj0,
        n_iter=100,
        verbose=True
    )

    
    # plot_traj_and_fieldmap(
    #     eif_table,
    #     sdf_field,
    #     map2d,
    #     traj0,
    #     traj_opt,
    #     traj_sdfonly_opt,
    #     save_dir="results",
    #     frame="traj_eif_sdf.pdf"
    # )

    plot_traj_and_fieldmap(
        eif_table,
        sdf_field,
        map2d,
        traj0,
        traj_opt,
        traj_sdfonly_opt,
        frame="traj_eif.pdf",
        show_sdf=False
    )


if __name__ == "__main__":
    main()
