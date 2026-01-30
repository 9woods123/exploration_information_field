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
                 lambda_info=0.25,
                 lambda_col=0.25,
                 mu_smooth=0.75,
                 step_size=0.05,
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

    # --------------------------------------------------
    # Main optimization loop
    # --------------------------------------------------
    def optimize(self, traj, n_iter=50, verbose=True):

        timer=Timer()

        traj = traj.copy()

        for it in range(n_iter):
            grad = self.total_gradient(traj)

            # ---- enforce boundary by zeroing gradient ----
            if self.fix_start:
                grad[0] = 0.0
            if self.fix_goal:
                grad[-1] = 0.0
            # print("grad[0]:",grad[0])

            traj.waypoints = traj.waypoints  - self.step_size * grad

            if verbose and it % 10 == 0:
                J = self.total_cost(traj)
                print(f"[TrajOpt] iter {it:03d}, cost = {J:.3f}")

        self.define_the_yaw(traj)

        timer.lap("Traj optimize ")


        return traj

    def define_the_yaw(self, traj_opt):

        wps = traj_opt.waypoints        # (N, 2)
        traj_opt.yaws = np.array([
            self.eif.query_yaw(p)
            for p in wps
        ])

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
        return cost * traj.dt

    def collision_cost(self, traj, eps=0.3):
        cost = 0.0
        for x in traj.waypoints:
            d = self.sdf.query(x)
            if d < eps:
                cost += (eps - d)**2
        return cost * traj.dt

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
        
        print(g)
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
