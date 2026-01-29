

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import time

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self, msg):
        t1 = time.perf_counter()
        dt_ms = (t1 - self.t0) * 1000.0
        print(f"[TIMER] {msg:35s}: {dt_ms:8.2f} ms")
        self.t0 = t1

# ============================================================
# Map
# ============================================================

# ============================================================
# SDF 2D (Signed Distance Field)
# ============================================================
class SDF2D:
    def __init__(self, map2d, xs, ys, resolution):
        self.map = map2d
        self.xs = xs
        self.ys = ys
        self.resolution = resolution

        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]

        self.sdf = np.zeros((len(xs), len(ys)))

    def build(self):
        """
        Build SDF grid:
        - known free (p < 0.1): positive
        - unknown / occupied: negative
        """
        free_pts = []
        obs_pts  = []

        for idx, p in self.map.grid.items():
            wp = self.map.grid_to_world(idx)
            if p ==0 :
                free_pts.append(wp)
            else:
                obs_pts.append(wp)

        free_pts = np.array(free_pts)
        obs_pts  = np.array(obs_pts)

        for ix, x in enumerate(self.xs):
            for iy, y in enumerate(self.ys):
                t = np.array([x, y])
                p = self.map.grid[self.map.world_to_grid(t)]

                is_obstacle = (p >= 0.1)

                if is_obstacle:
                    d = np.min(np.linalg.norm(free_pts - t, axis=1))
                    self.sdf[ix, iy] = -d
                else:
                    d = np.min(np.linalg.norm(obs_pts - t, axis=1))
                    self.sdf[ix, iy] = d

    def _bilinear(self, grid, x, y):
        ix = (x - self.xs[0]) / self.dx
        iy = (y - self.ys[0]) / self.dy

        i0 = int(np.floor(ix))
        j0 = int(np.floor(iy))

        if i0 < 0 or j0 < 0 or \
           i0 >= len(self.xs)-1 or j0 >= len(self.ys)-1:
            return 0.0

        tx = ix - i0
        ty = iy - j0

        v00 = grid[i0,   j0]
        v10 = grid[i0+1, j0]
        v01 = grid[i0,   j0+1]
        v11 = grid[i0+1, j0+1]

        return ((1-tx)*(1-ty)*v00 +
                tx*(1-ty)*v10 +
                (1-tx)*ty*v01 +
                tx*ty*v11)

    def query(self, t):
        """Query SDF value at continuous position t"""
        return self._bilinear(self.sdf, t[0], t[1])

    def grad(self, t, eps=0.5):
        """Numerical gradient of SDF"""
        ex = np.array([eps, 0])
        ey = np.array([0, eps])

        gx = (self.query(t + ex) - self.query(t - ex)) / (2*eps)
        gy = (self.query(t + ey) - self.query(t - ey)) / (2*eps)

        return np.array([gx, gy])
    


class Map2D:
    def __init__(self, resolution=0.2):
        self.resolution = resolution
        self.grid = defaultdict(lambda: 0.5)
        self.known = []
        self.unknown = []

    def world_to_grid(self, p):
        return (
            int(np.floor(p[0] / self.resolution)),
            int(np.floor(p[1] / self.resolution))
        )

    def grid_to_world(self, idx):
        return np.array([
            (idx[0] + 0.5) * self.resolution,
            (idx[1] + 0.5) * self.resolution
        ])

    def entropy(self, p):
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        return -p * np.log(p) - (1 - p) * np.log(1 - p)

    def init_circle_known(self, R, bound):
        for ix in range(-bound, bound):
            for iy in range(-bound, bound):
                d = np.hypot(ix, iy) * self.resolution
                if d < R:
                    self.grid[(ix, iy)] = 0.0
                    self.known.append((ix, iy))
                else:
                    self.grid[(ix, iy)] = 0.5
                    self.unknown.append((ix, iy))
               
    def is_free(self, p):
        idx = self.world_to_grid(p)
        return self.grid[idx] < 0.1
    
    def init_rectangle_known(self, center, width, height, bound):
        """
        初始化一个轴对齐的长方形已知区域

        center: (cx, cy) in world coordinates [m]
        width, height: rectangle size [m]
        bound_m: map half-size in meters (map spans [-bound_m, bound_m])
        """

        cx, cy = center
        hw = width * 0.5
        hh = height * 0.5

        # meters -> grid cells
        bound_cells = int(np.ceil(bound / self.resolution))

        self.known.clear()
        self.unknown.clear()

        for ix in range(-bound_cells, bound_cells):
            for iy in range(-bound_cells, bound_cells):

                p = self.grid_to_world((ix, iy))

                if (abs(p[0] - cx) <= hw) and (abs(p[1] - cy) <= hh):
                    self.grid[(ix, iy)] = 0.0   # free / known
                    self.known.append((ix, iy))
                else:
                    self.grid[(ix, iy)] = 0.5   # unknown
                    self.unknown.append((ix, iy))


    def entropy_at(self, p):
        return self.entropy(self.grid[self.world_to_grid(p)])


# ============================================================
# Sensor Model
# ============================================================
class SensorModel:
    def __init__(self, alpha, kf, kr, dmax):
        self.alpha = alpha
        self.kf = kf
        self.kr = kr
        self.dmax = dmax

    def v_range(self, d):
        return 1.0 / (1.0 + np.exp(self.kr * (d - self.dmax)))

    def v_fov(self, r, yaw):
        z = np.array([np.cos(yaw), np.sin(yaw)])
        cos_theta = np.dot(r, z) / (np.linalg.norm(r) + 1e-6)
        return 1.0 / (1.0 + np.exp(-self.kf * (cos_theta - np.cos(self.alpha))))


# ============================================================
# Info Point Sampler
# ============================================================
class InfoSampler:
    def __init__(self, map2d, sensor, H_thresh):
        self.map = map2d
        self.sensor = sensor
        self.H_thresh = H_thresh

    def sample(self, t, n):
        pts, w = [], []
        for _ in range(n):
            r = self.sensor.dmax * np.sqrt(np.random.rand())
            th = 2 * np.pi * np.random.rand()
            p = t + np.array([r*np.cos(th), r*np.sin(th)])

            H = self.map.entropy_at(p)
            if H < self.H_thresh:
                continue

            pts.append(p)
            w.append(H)
        return np.array(pts), np.array(w)


# ============================================================
# EIF Evaluator
# ============================================================
class EIFEvaluator:
    def __init__(self, sensor):
        self.sensor = sensor

    def I(self, t, yaw, pts, w):
        I = 0.0
        for pj, Hj in zip(pts, w):
            r = pj - t
            d = np.linalg.norm(r)
            I += Hj * self.sensor.v_range(d) * self.sensor.v_fov(r, yaw)
        return I

    def optimal_yaw_fast(self, t, pts, w):
        d = np.zeros(2)
        for pj, Hj in zip(pts, w):
            r = pj - t
            norm = np.linalg.norm(r) + 1e-6
            d += Hj * self.sensor.v_range(norm) * (r / norm)

        yaw = np.arctan2(d[1], d[0])
        I = self.I(t, yaw, pts, w)
        return yaw, I


# ============================================================
# KDE Continuous Field
# ============================================================
class KDEField:
    def __init__(self, ts, Is, h):
        self.ts = ts
        self.Is = Is
        self.h = h

    def eval(self, t):
        diff = self.ts - t
        r2 = np.sum(diff**2, axis=1)
        w = np.exp(-0.5 * r2 / (self.h**2))
        return np.sum(w * self.Is) / (np.sum(w) + 1e-6)


# ============================================================
# Gradient Estimator
# ============================================================
class GradientEstimator:
    def __init__(self, field, eps):
        self.field = field
        self.eps = eps

    def grad(self, t):
        ex = np.array([self.eps, 0])
        ey = np.array([0, self.eps])
        gx = (self.field.eval(t + ex) - self.field.eval(t - ex)) / (2*self.eps)
        gy = (self.field.eval(t + ey) - self.field.eval(t - ey)) / (2*self.eps)
        return np.array([gx, gy])


# ============================================================
# EIF Lookup Table (FAST QUERY)
# ============================================================
class EIFLookupTable:
    def __init__(self, xs, ys, I, Gx, Gy):
        self.xs = xs
        self.ys = ys
        self.I = I
        self.Gx = Gx
        self.Gy = Gy
        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]

    def _bilinear(self, grid, x, y):
        ix = (x - self.xs[0]) / self.dx
        iy = (y - self.ys[0]) / self.dy

        i0 = int(np.floor(ix))
        j0 = int(np.floor(iy))

        if i0 < 0 or j0 < 0 or \
           i0 >= len(self.xs)-1 or j0 >= len(self.ys)-1:
            return 0.0

        tx = ix - i0
        ty = iy - j0

        v00 = grid[i0,   j0]
        v10 = grid[i0+1, j0]
        v01 = grid[i0,   j0+1]
        v11 = grid[i0+1, j0+1]

        return ((1-tx)*(1-ty)*v00 +
                tx*(1-ty)*v10 +
                (1-tx)*ty*v01 +
                tx*ty*v11)

    def query_I(self, t):
        return self._bilinear(self.I, t[0], t[1])

    def query_grad(self, t):
        gx = self._bilinear(self.Gx, t[0], t[1])
        gy = self._bilinear(self.Gy, t[0], t[1])
        return np.array([gx, gy])


# ============================================================
# MAIN
# ============================================================
def main():
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

    sdf_t_test = np.array([4, 5])
    print("sdf_field():",sdf_field.query(sdf_t_test))
    print("sdf_field()grad:",sdf_field.grad(sdf_t_test))

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


    # -------------------------
    # Visualization
    # -------------------------
    # =====================================================
    # Visualization (EIF + SDF)
    # =====================================================
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    unknown_xy = np.array([map2d.grid_to_world(idx) for idx in map2d.unknown])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -------- EIF --------
    ax = axes[0]
    ax.scatter(unknown_xy[:,0], unknown_xy[:,1],
               s=5, c='lightgray', alpha=0.4)
    c1 = ax.contourf(X, Y, I_grid, 30, cmap='viridis')
    ax.quiver(X, Y, Gx_grid, Gy_grid,
              color='red', alpha=0.7, scale=300)
    ax.set_aspect('equal')
    ax.set_title("EIF Field + Gradient")
    fig.colorbar(c1, ax=ax, shrink=0.8)

    # -------- SDF --------
    ax = axes[1]
    c2 = ax.contourf(X, Y, sdf_field.sdf, 40, cmap='coolwarm')
    ax.contour(X, Y, sdf_field.sdf, levels=[0.0],
               colors='black', linewidths=2)
    ax.set_aspect('equal')
    ax.set_title("Signed Distance Field (Obstacle < 0)")
    fig.colorbar(c2, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
