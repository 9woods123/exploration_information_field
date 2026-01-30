import numpy as np
import matplotlib.pyplot as plt
from eif_map import *

class Trajectory:
    """
    Minimal trajectory: discrete-time 2D trajectory
    """

    def __init__(self, waypoints, dt):
        """
        waypoints: (N, 2) numpy array
        dt: time step
        """
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.dt = float(dt)
        self.N = self.waypoints.shape[0]
        self.yaws = None

        assert self.waypoints.ndim == 2
        assert self.waypoints.shape[1] == 2, "Trajectory must be 2D"

    def copy(self):
        return Trajectory(self.waypoints.copy(), self.dt)

    def __len__(self):
        return self.N

    def sample(self):
        """Return raw waypoints"""
        return self.waypoints




class TrajOpti:
    """
    Simple gradient-based trajectory optimizer
    """

    def __init__(self,
                 eif_table,
                 sdf,
                 lambda_info=0.0,
                 lambda_col=0.5,
                 mu_smooth=1,
                 step_size=0.1,
                 fix_start=True,
                 fix_goal=True):
        """
        eif_table : EIFLookupTable
        sdf       : SDFField (must provide eval(x), grad(x))
        """
        self.eif = eif_table
        self.sdf = sdf

        self.lambda_info=lambda_info
        self.lambda_col = lambda_col
        self.mu_smooth  = mu_smooth
        self.step_size  = step_size

        self.fix_start = fix_start
        self.fix_goal  = fix_goal

    def set_eif_table(self, eif_table):
        self.eif=eif_table

    def set_sd_table(self, sdf):
        self.sdf=sdf

    # # --------------------------------------------------
    # # Main optimization loop
    # # --------------------------------------------------
    # def optimize(self, traj, n_iter=50, verbose=True):

    #     timer=Timer()

    #     traj = traj.copy()

    #     for it in range(n_iter):

    #         grad = self.total_gradient(traj)

    #         # ---- enforce boundary by zeroing gradient ----
    #         if self.fix_start:
    #             grad[0] = 0.0
    #         if self.fix_goal:
    #             grad[-1] = 0.0
    #         # print("grad[0]:",grad[0])

    #         traj.waypoints = traj.waypoints  - self.step_size * grad

    #         # if verbose and it % 10 == 0:
    #         #     J = self.total_cost(traj)
    #         #     print(f"[TrajOpt] iter {it:03d}, cost = {J:.3f}")

    #     self.define_the_yaw(traj)

    #     timer.lap("Traj optimize ")


    #     return traj

    def optimize(self, traj, n_iter=50, verbose=True):

        timer = Timer()

        opt_traj = traj.copy()
        N = len(opt_traj.waypoints)

        for it in range(n_iter):

            for i in range(2, N - 2):

                # --- use UPDATED trajectory ---
                xim2 = opt_traj.waypoints[i - 2]
                xim1 = opt_traj.waypoints[i - 1]
                xi   = opt_traj.waypoints[i]
                xip1 = opt_traj.waypoints[i + 1]
                xip2 = opt_traj.waypoints[i + 2]

                smooth_grad, smooth_cost = self.smoothness_term(
                    xim2, xim1, xi, xip1, xip2
                )

                collision_grad, collision_cost = self.obstacle_term(xi)

                info_grad, info_cost=self.info_term(xi)

                correction = (
                      self.mu_smooth * smooth_grad
                    + self.lambda_col * collision_grad
                    + self.lambda_info * info_grad
                )

                opt_xi= xi - self.step_size * correction

                # --- update IN PLACE ---
                opt_traj.waypoints[i] = opt_xi

                
            # if verbose and it % 10 == 0:
            #     print(f"[TrajOpt] iter {it:03d}")

        self.define_the_yaw(opt_traj)
        timer.lap("Traj optimize")

        return opt_traj





    def define_the_yaw(self, traj_opt, eps=1):
        """
        Define yaw safely:
        1) use EIF yaw if information exists
        2) otherwise follow motion direction
        3) fallback to previous yaw
        """

        wps = traj_opt.waypoints      # (N, 2)
        N = len(wps)

        yaws = np.zeros(N)

        prev_yaw = 0.0

        for i in range(N):

            p = wps[i]

            # ---------- 1. EIF yaw ----------
            I = self.eif.query_I(p)
            yaw_eif=self.eif.query_yaw(p)


            if I > eps:
                yaw = yaw_eif

            else:
                # ---------- 2. motion direction ----------
                if i < N - 1:
                    v = wps[i + 1] - wps[i]
                else:
                    v = wps[i] - wps[i - 1]

                yaw = np.arctan2(v[1], v[0])


            yaws[i] = yaw

        traj_opt.yaws = yaws


    # --------------------------------------------------
    # Cost
    # --------------------------------------------------
    def total_cost(self, traj):
        
        uni_weight=self.lambda_info+self.lambda_col+self.mu_smooth

        return (
             self.lambda_info/uni_weight *self.info_cost(traj)
            + self.lambda_col/uni_weight * self.collision_cost(traj)
            + self.mu_smooth/uni_weight * self.smoothness_cost(traj)
        )

    def info_cost(self, traj):
        cost = 0.0
        for x in traj.waypoints:
            cost -= self.eif.query_I(x)
        return cost 

    def collision_cost(self, traj, eps=0.3):
        cost = 0.0
        for x in traj.waypoints:
            d = self.sdf.query(x)
            if d < eps:
                cost += (eps - d)**2
        return cost

    def smoothness_cost(self, traj):
        x = traj.waypoints
        acc = x[:-2] - 2*x[1:-1] + x[2:]
        return np.sum(np.linalg.norm(acc, axis=1)**2)

    # --------------------------------------------------
    # Gradient
    # --------------------------------------------------
    def total_gradient(self, traj):
        g_info = self.info_grad(traj)
        g_col  = self.collision_grad(traj)
        g_smo  = self.smoothness_grad(traj)


        uni_weight=self.lambda_info+self.lambda_col+self.mu_smooth

        return (
            self.lambda_info/uni_weight   *  g_info
            + self.lambda_col/uni_weight  *  g_col
            + self.mu_smooth/uni_weight   *  g_smo
        )


    def info_grad(self, traj):
        """
        ∂ / ∂x [ -I(x) ]
        """
        g = np.zeros_like(traj.waypoints)
        for i, x in enumerate(traj.waypoints):
            g[i] = - self.eif.query_grad(x)
        
        return g 


    def collision_grad(self, traj, eps=1.5):
        """
        Soft collision avoidance gradient:
        Penalize sdf < eps
        """
        g = np.zeros_like(traj.waypoints)

        for i, x in enumerate(traj.waypoints):
            d = self.sdf.query(x)          # ← 正确

            if d < eps:
                g[i] = -2.0 * (eps - d) * self.sdf.grad(x)
                # print("x:",x)
                # print("d:", d)
                # print("self.sdf.grad(x):", self.sdf.grad(x))
                

        return g 



    def smoothness_grad(self, traj):
        """
        Gradient of sum ||x_{k-1} - 2x_k + x_{k+1}||^2
        """
        x = traj.waypoints
        N = traj.N
        g = np.zeros_like(x)


        # print("x:", x)
        # print("g:", g)

        for k in range(2, N - 2):
            g[k] = 2 * (
                x[k-2]
                - 4*x[k-1]
                + 6*x[k]
                - 4*x[k+1]
                + x[k+2]
            )
        
        # print("g:",g)

        return g
        
    def smoothness_term(self, xim2, xim1, xi, xip1, xip2):
        """
        5-point smoothness term
        """

        grad = 2.0 * (
            xim2
            - 4.0 * xim1
            + 6.0 * xi
            - 4.0 * xip1
            + xip2
        )

        # cost（可选，仅用于调试 / 监控）
        acc = xim1 - 2.0 * xi + xip1
        cost = np.dot(acc, acc)

        return grad, cost


    def obstacle_term(self, xi, eps=2.5):
        """
        Soft obstacle avoidance (point-wise)
        """

        grad = np.zeros_like(xi)
        cost = 0.0

        d = self.sdf.query(xi)

        if d < eps:
            grad = -2.0 * (eps - d) * self.sdf.grad(xi)
            cost = (eps - d)**2

        return grad, cost
                    
    def info_term(self, xi):
        """
        ∂ / ∂x [ -I(x) ]
        """

        g = - self.eif.query_grad(xi)
        cost = - self.eif.query_I(xi)

        return g 


class PathPlanner:
    """
    Minimal path planner:
    straight-line / polyline initialization + trajectory optimization
    """

    def __init__(self, 
                 n_waypoints=10,
                 dt=0.2):
        self.n_waypoints = n_waypoints
        self.dt          = dt

    # -------------------------------------------------
    # Straight-line initialization
    # -------------------------------------------------
    def init_straight_traj(self, start, goal):
        start = np.asarray(start)
        goal  = np.asarray(goal)

        waypoints = np.linspace(start, goal, self.n_waypoints)
        return Trajectory(waypoints, self.dt)

    # -------------------------------------------------
    # Polyline initialization (3 points)
    # -------------------------------------------------
    def init_polyline_traj(self, start, mid, goal):
        """
        Initialize a 2-segment polyline trajectory:
        start -> mid -> goal
        """
        start = np.asarray(start)
        mid   = np.asarray(mid)
        goal  = np.asarray(goal)

        # --- segment lengths ---
        L1 = np.linalg.norm(mid - start)
        L2 = np.linalg.norm(goal - mid)
        L  = L1 + L2 + 1e-8

        # --- allocate waypoints proportionally ---
        n1 = max(2, int(self.n_waypoints * L1 / L))
        n2 = self.n_waypoints - n1 + 1  # +1 because mid is shared

        # --- build segments ---
        seg1 = np.linspace(start, mid, n1)
        seg2 = np.linspace(mid, goal, n2)[1:]  # remove duplicated mid

        waypoints = np.vstack([seg1, seg2])

        # safety check
        if len(waypoints) != self.n_waypoints:
            raise ValueError("Waypoint count mismatch")

        return Trajectory(waypoints, self.dt)
