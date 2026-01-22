


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# =========================
# 计时工具
# =========================
def tic():
    return time.perf_counter()

def toc(t0, name=""):
    dt = time.perf_counter() - t0
    print(f"[TIME] {name:<35s}: {dt*1000:7.2f} ms")
    return dt

# =========================
# 参数
# =========================
resolution = 0.2
H_THRESH = 0.65 * np.log(2)

ALPHA = np.pi / 2        # FoV 半角
KF = 10.0
KR = 4.0
D_MAX = 5.0

NUM_T_SAMPLES = 100
KERNEL_H = 0.8
MAX_INFO_PTS = 100

R_EVAL = 7.0
STEP = 0.2

yaw = 3*3.14/4

# =========================
# 地图（hashmap）
# =========================
grid = defaultdict(lambda: 0.5)


def grid_to_world(idx):
    return np.array([
        (idx[0] + 0.5) * resolution,
        (idx[1] + 0.5) * resolution
    ])

known_cells = []
unknown_cells = []

for ix in range(-50, 50):
    for iy in range(-50, 50):
        d = np.hypot(ix, iy) * resolution
        if d < 7.0:
            grid[(ix, iy)] = 0.0
            known_cells.append((ix, iy))
        else:
            grid[(ix, iy)] = 0.5
            unknown_cells.append((ix, iy))

# =========================
# 熵
# =========================
def entropy(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def extract_info_cells(grid):
    pts, w = [], []
    for idx, p in grid.items():
        H = entropy(p)
        if H > H_THRESH:
            pts.append(idx)
            w.append(H)
    return pts, np.array(w)

# =========================
# 传感器模型
# =========================
def v_range(d):
    return 1.0 / (1.0 + np.exp(KR * (d - D_MAX)))

def v_fov(r, yaw):
    z = np.array([np.cos(yaw), np.sin(yaw)])
    cos_theta = np.dot(r, z) / (np.linalg.norm(r) + 1e-6)
    return 1.0 / (1.0 + np.exp(-KF * (cos_theta - np.cos(ALPHA))))

# =========================
# EIF 标量 I(t)
# =========================
def compute_I(t, yaw, info_pts, info_w):
    I = 0.0
    for idx, w in zip(info_pts, info_w):
        pj = grid_to_world(idx)
        r = pj - t
        d = np.linalg.norm(r)
        I += w * v_range(d) * v_fov(r, yaw)
    return I

# =========================
# 采样
# =========================
def sample_known_t(n):
    sel = np.random.choice(len(known_cells), n, replace=False)
    return np.array([grid_to_world(known_cells[i]) for i in sel])




def sample_info_pts(pts, w, max_n):
    if len(pts) <= max_n:
        return pts, w
    sel = np.random.choice(len(pts), max_n, replace=False)
    return [pts[i] for i in sel], w[sel]



def sample_info_points_local(
    t,
    R=D_MAX,
    n=50,
    w_mean=1,
    w_std=0
):
    
        
    """
    在 t 周围半径 R 内采样 info points
    权重 w = H(grid(p))
    """
    pts = []
    w = []

    for _ in range(n):
        r = R * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()

        p = t + np.array([
            r * np.cos(theta),
            r * np.sin(theta)
        ])

        # --- 查询地图 ---
        ix = int(np.floor(p[0] / resolution))
        iy = int(np.floor(p[1] / resolution))

        p_occ = grid[(ix, iy)] 
        H_p = entropy(p_occ)

        if H_p < H_THRESH:
            continue   # 已知区域，直接丢掉（重要）

        pts.append(p)
        w.append(H_p)

    return np.array(pts), np.array(w)



def sample_infopoints_around_viewpoint(
    t,
    R=D_MAX,
    n=MAX_INFO_PTS,
    w_mean=0,
    w_std=0.0
):
    """
    给定一个 viewpoint t
    在其周围采样 info points
    """
    info_pts_t, info_w_t = sample_info_points_local(
        t,
        R=R,
        n=n,
        w_mean=w_mean,
        w_std=w_std
    )
    return info_pts_t, info_w_t

    
def compute_I_continuous(t, yaw, info_pts, info_w):
    I = 0.0
    for pj, w in zip(info_pts, info_w):
        r = pj - t
        d = np.linalg.norm(r)
        I += w * v_range(d) * v_fov(r, yaw)
    return I



# =========================
# Kernel Regression (KDE)
# =========================
def kernel_I(t, ts, Is, h):
    diff = ts - t
    r2 = np.sum(diff**2, axis=1)
    w = np.exp(-0.5 * r2 / (h**2))
    return np.sum(w * Is) / (np.sum(w) + 1e-6)




#1
t0 = tic()
ts = sample_known_t(NUM_T_SAMPLES)
toc(t0, "sample_known_t")



t0 = tic()
Is = []

for t in ts:
    info_pts_t, info_w_t = sample_infopoints_around_viewpoint(t)
    I_t = compute_I_continuous(t, yaw, info_pts_t, info_w_t)
    Is.append(I_t)

Is = np.array(Is)
toc(t0, "compute_I ")

# 4️⃣ KDE 连续场（圆形）
t0 = tic()
X, Y, Z = [], [], []

xs = np.arange(-R_EVAL, R_EVAL + STEP, STEP)
ys = np.arange(-R_EVAL, R_EVAL + STEP, STEP)

for x in xs:
    for y in ys:
        if x*x + y*y > R_EVAL*R_EVAL:
            continue
        t = np.array([x, y])
        I = kernel_I(t, ts, Is, KERNEL_H)
        X.append(x)
        Y.append(y)
        Z.append(I)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

toc(t0, "kernel regression field eval")

# 5️⃣ 可视化
t0 = tic()

plt.figure(figsize=(7, 7))
plt.tricontourf(X, Y, Z, levels=30, cmap='viridis')

unknown = np.array([grid_to_world(idx) for idx in unknown_cells])
plt.scatter(unknown[:, 0], unknown[:, 1],
            s=5, c='lightgray', alpha=0.9, label='unknown')

plt.scatter(ts[:, 0], ts[:, 1],
            s=25, c='red', marker='x', label='sampled t')

plt.scatter(0, 0, c='white', s=80,
            edgecolors='black', label='robot')

plt.colorbar(label='Information Value')
plt.axis('equal')
plt.legend(loc='upper right')
plt.title("Continuous EIF via Kernel Regression (Single Frame)")
plt.show()

toc(t0, "visualization")

