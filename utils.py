import math
import numpy as np
from matplotlib.patches import Wedge

def circular_mean(yaws, weights):
    sin_sum = np.sum(weights * np.sin(yaws))
    cos_sum = np.sum(weights * np.cos(yaws))
    return np.arctan2(sin_sum, cos_sum)


def draw_yaw_arrows(ax, traj_pts, yaws, step=1, scale=0.4, color='orange'):
    """
    在轨迹上画 yaw 方向箭头
    """
    xs = traj_pts[::step, 0]
    ys = traj_pts[::step, 1]
    us = np.cos(yaws[::step])
    vs = np.sin(yaws[::step])

    ax.quiver(
        xs, ys,
        us, vs,
        angles='xy',
        scale_units='xy',
        scale=1.0 / scale,
        color=color,
        width=0.008,
        zorder=6
    )

def draw_fov_wedges(
    ax,
    traj_pts,
    yaws,
    alpha=np.pi/3,     # half FoV angle
    r=0.8,             # FoV range
    step=3,
    color='orange',
    alpha_fill=0.6
):
    """
    画相机 FoV 扇形（实心三角扇形）
    """
    for p, yaw in zip(traj_pts[::step], yaws[::step]):
        theta1 = np.degrees(yaw - alpha)
        theta2 = np.degrees(yaw + alpha)

        wedge = Wedge(
            center=(p[0], p[1]),
            r=r,
            theta1=theta1,
            theta2=theta2,
            facecolor=color,
            edgecolor='none',
            alpha=alpha_fill,
            zorder=5
        )
        ax.add_patch(wedge)
