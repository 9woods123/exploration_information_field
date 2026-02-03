import numpy as np
import matplotlib.pyplot as plt

def generate_dense_maze(K=5, cell_size=2.0, wall_thickness=0.5, resolution=0.2, seed=42):
    """
    Kruskal生成迷宫并映射到稠密栅格
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
            if j < K-1: edges.append((i,j,0))
            if i < K-1: edges.append((i,j,1))
    np.random.shuffle(edges)

    for i,j,flag in edges:
        xy = i*K + j
        nxy = xy + (1 if flag==0 else K)
        f1,f2 = findftr(xy), findftr(nxy)
        if f1 != f2:
            ftr[f1] = f2
            maze[2*i+1+flag, 2*j+2-flag] = 0

    # ----------------------
    # 映射到稠密 grid_map
    # ----------------------
    # 每个maze单元格占 grid_cell_count 个稠密格子
    grid_cell_count = int(np.ceil(cell_size / resolution))
    wall_cell_count = int(np.ceil(wall_thickness / resolution))

    grid_size = M*grid_cell_count
    grid_map = np.full((grid_size, grid_size), 0.5)  # unknown

    for ix in range(M):
        for iy in range(M):
            val = maze[ix, iy]
            # 确定稠密网格范围
            gx_start = ix*grid_cell_count - wall_cell_count//2
            gx_end   = ix*grid_cell_count + grid_cell_count + wall_cell_count//2
            gy_start = iy*grid_cell_count - wall_cell_count//2
            gy_end   = iy*grid_cell_count + grid_cell_count + wall_cell_count//2
            gx_start = max(0, gx_start)
            gy_start = max(0, gy_start)
            gx_end = min(grid_size, gx_end)
            gy_end = min(grid_size, gy_end)

            grid_map[gx_start:gx_end, gy_start:gy_end] = val

    return grid_map

# --------------------------
# 可视化
# --------------------------
grid_map = generate_dense_maze(K=3, cell_size=2.0, wall_thickness=0.5, resolution=0.2, seed=123)

plt.figure(figsize=(8,8))
plt.imshow(grid_map.T[::-1,:], cmap='gray', origin='lower', vmin=0, vmax=1)
plt.title("Kruskal Maze - Continuous Walls")
plt.show()
