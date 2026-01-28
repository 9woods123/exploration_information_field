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

        assert self.waypoints.ndim == 2
        assert self.waypoints.shape[1] == 2, "Trajectory must be 2D"

    def copy(self):
        return Trajectory(self.waypoints.copy(), self.dt)

    def __len__(self):
        return self.N

    def sample(self):
        """Return raw waypoints"""
        return self.waypoints

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.waypoints[:, 0], self.waypoints[:, 1], '-o', **kwargs)
        ax.scatter(self.waypoints[0, 0],  self.waypoints[0, 1],
                   c='g', s=80, label='start')
        ax.scatter(self.waypoints[-1, 0], self.waypoints[-1, 1],
                   c='r', s=80, label='end')
        ax.axis('equal')
        ax.legend()
        return ax




class TrajOpti:
    """
    Simple gradient-based trajectory optimizer
    """

    def __init__(self,
                 eif_table,
                 sdf,
                 lambda_col=1.0,
                 mu_smooth=1.0,
                 step_size=0.1,
                 fix_start=True,
                 fix_goal=True):
        """
        eif_table : EIFLookupTable
        sdf       : SDFField (must provide eval(x), grad(x))
        """
        self.eif = eif_table
        self.sdf = sdf

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
        traj = traj.copy()

        for it in range(n_iter):
            grad = self.total_gradient(traj)

            # ---- enforce boundary by zeroing gradient ----
            if self.fix_start:
                grad[0] = 0.0
            if self.fix_goal:
                grad[-1] = 0.0

            traj.waypoints = traj.waypoints  - self.step_size * grad

            if verbose and it % 10 == 0:
                J = self.total_cost(traj)
                print(f"[TrajOpt] iter {it:03d}, cost = {J:.3f}")

        return traj


    # --------------------------------------------------
    # Cost
    # --------------------------------------------------
    def total_cost(self, traj):
        return (
            self.info_cost(traj)
            + self.lambda_col * self.collision_cost(traj)
            + self.mu_smooth * self.smoothness_cost(traj)
        )

    def info_cost(self, traj):
        cost = 0.0
        for x in traj.waypoints:
            cost -= self.eif.query_I(x)
        return cost * traj.dt

    def collision_cost(self, traj, eps=0.3):
        cost = 0.0
        for x in traj.waypoints:
            d = self.sdf.eval(x)
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

        return (
            g_info
            + self.lambda_col * g_col
            + self.mu_smooth * g_smo
        )

    def info_grad(self, traj):
        """
        ∂ / ∂x [ -I(x) ]
        """
        g = np.zeros_like(traj.waypoints)
        for i, x in enumerate(traj.waypoints):
            g[i] = -self.eif.query_grad(x)
        return g * traj.dt

    def collision_grad(self, traj, eps=0.75):
        g = np.zeros_like(traj.waypoints)
        for i, x in enumerate(traj.waypoints):
            d = self.sdf.eval(x)
            if d < eps:
                g[i] = -2 * (eps - d) * self.sdf.grad(x)
        return g * traj.dt


    def smoothness_grad(self, traj):
        """
        Gradient of sum ||x_{k-1} - 2x_k + x_{k+1}||^2
        """
        x = traj.waypoints
        N = traj.N
        g = np.zeros_like(x)

        for k in range(2, N - 2):
            g[k] = 2 * (
                x[k-2]
                - 4*x[k-1]
                + 6*x[k]
                - 4*x[k+1]
                + x[k+2]
            )

        return g



class PathPlanner:
    """
    Minimal path planner:
    straight-line initialization + trajectory optimization
    """

    def __init__(self, 
                 n_waypoints=40,
                 dt=0.2):
        """
        traj_opti : TrajOpti instance
        n_waypoints: number of trajectory points
        dt         : time step
        """
        self.n_waypoints = n_waypoints
        self.dt          = dt

    # -------------------------------------------------
    # Straight-line initialization
    # -------------------------------------------------
    def init_straight_traj(self, start, goal):
        """
        start, goal: (2,)
        """
        start = np.asarray(start)
        goal  = np.asarray(goal)

        waypoints = np.linspace(start, goal, self.n_waypoints)
        return Trajectory(waypoints, self.dt)

    # -------------------------------------------------
    # Main planning interface
    # -------------------------------------------------
    def plan(self, start, goal,
             n_iter=60,
             verbose=True):
        """
        Returns optimized trajectory
        """

        # 1) initialize
        traj0 = self.init_straight_traj(start, goal)

        # 2) optimize
        traj_opt = self.traj_opti.optimize(
            traj0,
            n_iter=n_iter,
            verbose=verbose
        )

        return traj_opt
