

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import *
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
    
    def is_occupied(self, p):
        idx = self.world_to_grid(p)
        return self.grid[idx] ==1.0



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

    def add_random_rectangular_obstacles(
        self,
        n_obs=4,
        w_range=(0.5, 4),
        h_range=(0.5, 6),
        seed=22
    ):
        
        ## seed=0

        if seed is not None:
            np.random.seed(seed)

        known_world = np.array(
            [self.grid_to_world(idx) for idx in self.known]
        )

        for _ in range(n_obs):
            c = known_world[np.random.randint(len(known_world))]
            w = np.random.uniform(*w_range)
            h = np.random.uniform(*h_range)

            for idx in list(self.known):
                p = self.grid_to_world(idx)
                if abs(p[0] - c[0]) <= w/2 and abs(p[1] - c[1]) <= h/2:
                    self.grid[idx] = 1.0
    
    def init_T_corridor(
        self,
        center=(0.0, 0.0),
        w_vert=2.0,     # 竖直走廊宽度
        h_vert=10.0,    # 竖直走廊长度
        w_horiz=8.0,    # 水平走廊长度
        h_horiz=2.0,    # 水平走廊宽度
        wall_thickness=0.6,  # 走廊两侧墙体厚度
        bound=15.0      # 地图半尺寸（米）
    ):
        """
        初始化一个 T 型走廊环境：
        - 走廊内部：已知自由 (0.0)
        - 走廊墙体：已知障碍 (1.0)
        - 外部区域：未知 (0.5)
        """

        cx, cy = center
        bound_cells = int(np.ceil(bound / self.resolution))

        self.known.clear()
        self.unknown.clear()

        def in_vertical_corridor(p):
            return (
                abs(p[0] - cx) <= w_vert * 0.5 and
                abs(p[1] - cy) <= h_vert * 0.5
            )

        def in_horizontal_corridor(p):
            return (
                abs(p[0] - cx) <= w_horiz * 0.5 and
                abs(p[1] - (cy + h_vert * 0.5 - h_horiz * 0.5)) <= h_horiz * 0.5
            )

        def in_corridor(p):
            return in_vertical_corridor(p) or in_horizontal_corridor(p)

        def in_wall(p):
            # 仅竖直走廊两侧生成墙
            in_vert_wall = (
                abs(p[0] - cx) <= w_vert * 0.5 + wall_thickness and
                abs(p[1] - cy) <= h_vert * 0.5 and   # 注意这里没有+wall_thickness
                not in_vertical_corridor(p)
            )
            return in_vert_wall

        for ix in range(-bound_cells, bound_cells):
            for iy in range(-bound_cells, bound_cells):

                idx = (ix, iy)
                p = self.grid_to_world(idx)

                if in_corridor(p):
                    self.grid[idx] = 0.0
                    self.known.append(idx)

                elif in_wall(p):
                    self.grid[idx] = 1.0
                    self.known.append(idx)

                else:
                    self.grid[idx] = 0.5
                    self.unknown.append(idx)

    def init_dense_maze(self, K=3, cell_size=3.0, wall_thickness=0.5, seed=42):
        """
        在 Map2D 中生成稠密迷宫（Kruskal生成完美迷宫）
        - 自动填充 self.grid、self.known、self.unknown
        """
        np.random.seed(seed)
        M = 2*K + 1
        maze = np.ones((M, M), dtype=int)
        for i in range(K):
            for j in range(K):
                maze[2*i+1, 2*j+1] = 0  # 通道中心

        # 并查集
        ftr = np.arange(K*K)
        def findftr(x):
            if ftr[x] != x:
                ftr[x] = findftr(ftr[x])
            return ftr[x]

        # 边列表
        edges = []
        for i in range(K):
            for j in range(K):
                if j < K-1: edges.append((i,j,0))  # 右边
                if i < K-1: edges.append((i,j,1))  # 下边
        np.random.shuffle(edges)

        for i,j,flag in edges:
            xy = i*K + j
            nxy = xy + (1 if flag==0 else K)
            f1,f2 = findftr(xy), findftr(nxy)
            if f1 != f2:
                ftr[f1] = f2
                maze[2*i+1+flag, 2*j+2-flag] = 0

        # ----------------------
        # 填充 Map2D 网格
        # ----------------------
        self.known.clear()
        self.unknown.clear()

        grid_cell_count = int(np.ceil(cell_size / self.resolution))
        wall_cell_count = int(np.ceil(wall_thickness / self.resolution))
        grid_size = M*grid_cell_count

        offset = grid_size // 2
        for ix in range(M):
            for iy in range(M):
                val = float(maze[ix, iy])
                gx_start = ix*grid_cell_count - wall_cell_count//2
                gx_end   = ix*grid_cell_count + grid_cell_count + wall_cell_count//2
                gy_start = iy*grid_cell_count - wall_cell_count//2
                gy_end   = iy*grid_cell_count + grid_cell_count + wall_cell_count//2
                gx_start = max(0, gx_start)
                gy_start = max(0, gy_start)
                gx_end = min(grid_size, gx_end)
                gy_end = min(grid_size, gy_end)

                for gx in range(gx_start, gx_end):
                    for gy in range(gy_start, gy_end):
                        idx = (gx - offset, gy - offset)  # ← 平移到中心
                        self.grid[idx] = val
                        if val == 0.0 or val == 1.0:
                            if idx not in self.known:
                                self.known.append(idx)
                        else:
                            if idx not in self.unknown:
                                self.unknown.append(idx)



    def  add_continuous_unknown(self, centers=[(0.0,0.0)], radius=2.0):
        """
        在已知迷宫上添加连续未知区域（世界坐标输入）
        centers: list of (x, y) in world coordinates [m]
        radius: 控制未知区域半径 [m]
        """
        radius_cells = int(np.ceil(radius / self.resolution))  # 转换为栅格半径
        new_unknown = []

        for idx in self.known[:]:
            x_idx, y_idx = idx
            p = self.grid_to_world(idx)  # 当前格子世界坐标
            for cx, cy in centers:
                if np.hypot(p[0] - cx, p[1] - cy) <= radius:
                    self.grid[idx] = 0.5
                    if idx in self.known:
                        self.known.remove(idx)
                    new_unknown.append(idx)
                    break  # 已经变为未知，不用检测其他中心

        self.unknown.extend(new_unknown)


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


    # def visibility_sample(self, t, n):


    #     pts, w = [], []

    #     for _ in range(n):
    #         # sample in polar coordinates (sensor range)
    #         r = self.sensor.dmax * np.sqrt(np.random.rand())
    #         th = 2 * np.pi * np.random.rand()
    #         p = t + np.array([r*np.cos(th), r*np.sin(th)])


    #         # ray casting: check visibility
    #         if not self._is_visible(t, p):
    #             continue

    #         H = self.map.entropy_at(p)
            
    #         if H < self.H_thresh:
    #             continue

    #         pts.append(p)
    #         w.append(H)


    #     return np.array(pts), np.array(w)
    # def _is_visible(self, p0, p1, step=0.1):
    #     direction = p1 - p0
    #     dist = np.linalg.norm(direction)
    #     direction /= dist

    #     s = 0.0
    #     s+=step
    #     while s < dist:
    #         p = p0 + s * direction

    #         if self.map.is_occupied(p):
    #             return False

    #         s += step
        
    #     return True

    def visibility_sample(self, t, n_rays, step=0.25):
        """
        从位置 t 发射 n_rays 条均匀分布的射线，沿每条射线探测未知区域
        t: np.array([x, y]) 当前位姿
        n_rays: 射线数量
        step: 射线步长
        """
        pts, w = [], []

        angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)  # 均匀角度

        for th in angles:
            direction = np.array([np.cos(th), np.sin(th)])
            s = step

            while s < self.sensor.dmax:
                p = t + s * direction

                if self.map.is_occupied(p):
                    break  # 被遮挡，射线终止

                H = self.map.entropy_at(p)
                if H >= self.H_thresh:
                    pts.append(p)
                    w.append(H)
                    break  # 遇到未知区域，射线终止

                s += step

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
    def __init__(self, xs, ys, I, Gx, Gy, Yaw):
        self.xs = xs
        self.ys = ys
        self.I  = I
        self.Gx = Gx
        self.Gy = Gy
        self.Yaw = Yaw

        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0]

    # ---------------------------
    # Nearest-cell yaw query
    # ---------------------------
    def query_yaw(self, x):
        ix, iy = self._coord_to_index(x)
        return self.Yaw[ix, iy]

    def _coord_to_index(self, x):
        ix = int(round((x[0] - self.xs[0]) / self.dx))
        iy = int(round((x[1] - self.ys[0]) / self.dy))

        ix = np.clip(ix, 0, len(self.xs) - 1)
        iy = np.clip(iy, 0, len(self.ys) - 1)

        return ix, iy

    # ---------------------------
    # Bilinear for scalar fields
    # ---------------------------
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

    def query_I(self, x):
        return self._bilinear(self.I, x[0], x[1])

    def query_grad_raw(self, x):
        gx = self._bilinear(self.Gx, x[0], x[1])
        gy = self._bilinear(self.Gy, x[0], x[1])
        return np.array([gx, gy])


    def query_grad(
        self,
        x,
        g_max=1.0,
        normalize=True
    ):
        g = self.query_grad_raw(x)

        # --- clip ---
        n = np.linalg.norm(g)
        if n > g_max:
            g = g / (n + 1e-6) * g_max

        # --- direction only ---
        if normalize:
            g = g / (np.linalg.norm(g) + 1e-6)

        return g

# ============================================================
# MAIN
# ============================================================


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


    map2d.init_rectangle_known(
        center=(0.0, 0.0),
        width=10.0,
        height=15.0,
        bound=MAP_BOUND
    )

    map2d.add_random_rectangular_obstacles(
        n_obs=8,
        w_range=(0.5, 4),
        h_range=(0.5, 4),
        seed=221221
    )


    # map2d.init_dense_maze(K=4, cell_size=3.0, wall_thickness=0.5, seed=5)
    # map2d.add_continuous_unknown(centers=[(0,5),(4,-3.0)], radius=2)


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
    plot_eif_and_sdf(eif_table, sdf_field,eif_table.Yaw,map2d ,show_sdf=True)



if __name__ == "__main__":
    main()
