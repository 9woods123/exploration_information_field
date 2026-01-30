import math
import numpy as np
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42     # TrueType
mpl.rcParams['ps.fonttype']  = 42


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


def plot_traj_and_fieldmap(
    eif_table,
    sdf_field,
    map2d,
    traj0,
    traj_opt,
    traj_sdf_only_opt,
    save_dir="results",
    fname="traj_eif_sdf.pdf"
):
    xs = eif_table.xs
    ys = eif_table.ys
    I_grid = eif_table.I
    SDF = sdf_field.sdf

    X, Y = np.meshgrid(xs, ys, indexing='ij')

    unknown_xy = np.array(
        [map2d.grid_to_world(idx) for idx in map2d.unknown]
    )

    traj0_pts = traj0.waypoints
    traj_opt_pts = traj_opt.waypoints
    traj_sdf_pts = traj_sdf_only_opt.waypoints

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # =====================================================
    # LEFT: EIF FIELD + trajectories
    # =====================================================
    ax = axes[0]

    # unknown area
    if len(unknown_xy) > 0:
        ax.scatter(
            unknown_xy[:, 0],
            unknown_xy[:, 1],
            s=4,
            c='lightgray',
            alpha=0.5,
            label='unknown'
        )

    # EIF
    c1 = ax.contourf(
        X, Y, I_grid,
        levels=30,
        cmap='viridis'
    )
    fig.colorbar(c1, ax=ax, shrink=0.8, label="EIF")

    # ---- init traj ----
    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='pink',
        linewidth=1.8,
        label='init traj'
    )

    # ---- EIF + SDF optimized ----
    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='EIF + SDF opt'
    )

    # ---- SDF only optimized ----
    ax.plot(
        traj_sdf_pts[:, 0],
        traj_sdf_pts[:, 1],
        '-c',
        linewidth=2.2,
        label='SDF only opt'
    )

    # start / goal
    ax.scatter(
        traj0_pts[0, 0],
        traj0_pts[0, 1],
        c='lime',
        s=40,
        zorder=5,
        label='start'
    )
    ax.scatter(
        traj0_pts[-1, 0],
        traj0_pts[-1, 1],
        c='red',
        s=40,
        zorder=5,
        label='goal'
    )

    # FOV
    draw_fov_wedges(ax, traj_opt_pts, traj_opt.yaws)

    ax.set_title("EIF field + trajectories")
    ax.set_aspect('equal')
    ax.legend(
        loc='upper right',
        fontsize=6,
        framealpha=0.85
    )

    # =====================================================
    # RIGHT: SDF FIELD + trajectories
    # =====================================================
    ax = axes[1]

    c2 = ax.contourf(
        X, Y, SDF,
        levels=40,
        cmap='coolwarm'
    )
    fig.colorbar(c2, ax=ax, shrink=0.8, label="SDF")

    # obstacle boundary
    ax.contour(
        X, Y, SDF,
        levels=[0.0],
        colors='black',
        linewidths=2
    )

    # ---- init traj ----
    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='pink',
        linewidth=1.8,
        label='init traj'
    )

    # ---- EIF + SDF optimized ----
    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='EIF + SDF opt'
    )

    # ---- SDF only optimized ----
    ax.plot(
        traj_sdf_pts[:, 0],
        traj_sdf_pts[:, 1],
        '-c',
        linewidth=2.2,
        label='SDF only opt'
    )

    ax.scatter(
        traj0_pts[0, 0],
        traj0_pts[0, 1],
        c='lime',
        s=40,
        zorder=5,
        label='start'
    )
    ax.scatter(
        traj0_pts[-1, 0],
        traj0_pts[-1, 1],
        c='red',
        s=40,
        zorder=5,
        label='goal'
    )

    draw_fov_wedges(ax, traj_opt_pts, traj_opt.yaws)

    ax.set_title("SDF field + trajectories")
    ax.set_aspect('equal')
    ax.legend(
        loc='upper right',
        fontsize=6,
        framealpha=0.85
    )

    plt.tight_layout()

    # -----------------------------
    # save figure (IEEE-safe)
    # -----------------------------

    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Figure saved] {save_path}")

    plt.show()


def plot_eif_and_sdf_with_traj(eif_table, sdf_field, map2d, traj0, traj_opt):
    xs = eif_table.xs
    ys = eif_table.ys
    I_grid = eif_table.I
    SDF = sdf_field.sdf

    X, Y = np.meshgrid(xs, ys, indexing='ij')

    unknown_xy = np.array(
        [map2d.grid_to_world(idx) for idx in map2d.unknown]
    )

    traj0_pts = traj0.waypoints
    traj_opt_pts = traj_opt.waypoints

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # =====================================================
    # LEFT: EIF FIELD
    # =====================================================
    ax = axes[0]

    if len(unknown_xy) > 0:
        ax.scatter(
            unknown_xy[:, 0],
            unknown_xy[:, 1],
            s=5,
            c='lightgray',
            alpha=0.6,
            label='unknown'
        )

    c1 = ax.contourf(
        X, Y, I_grid,
        levels=30,
        cmap='viridis'
    )
    fig.colorbar(c1, ax=ax, shrink=0.8, label="EIF")

    # ---- init trajectory ----
    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='white',
        linewidth=2,
        label='init traj'
    )
    ax.scatter(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        c='white',
        s=25,
        edgecolors='black',
        linewidths=0.5,
        zorder=4,
        label='init waypoints'
    )

    # ---- optimized trajectory ----
    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='opt traj'
    )
    ax.scatter(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        c='red',
        s=30,
        edgecolors='black',
        linewidths=0.5,
        zorder=5,
        label='opt waypoints'
    )

    # start / goal
    ax.scatter(
        traj_opt_pts[0, 0],
        traj_opt_pts[0, 1],
        c='lime',
        s=80,
        zorder=6,
        label='start'
    )
    ax.scatter(
        traj_opt_pts[-1, 0],
        traj_opt_pts[-1, 1],
        c='red',
        s=80,
        zorder=6,
        label='goal'
    )


    draw_fov_wedges(
        ax,
        traj_opt_pts,
        traj_opt.yaws
    )

    ax.set_title("EIF + SDF ")
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7)

    # =====================================================
    # RIGHT: SDF FIELD
    # =====================================================
    ax = axes[1]

    c2 = ax.contourf(
        X, Y, SDF,
        levels=40,
        cmap='coolwarm'
    )
    fig.colorbar(c2, ax=ax, shrink=0.8, label="SDF")

    # zero level set = obstacle boundary
    ax.contour(
        X, Y, SDF,
        levels=[0.0],
        colors='black',
        linewidths=2
    )

    # ---- init trajectory ----
    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='white',
        linewidth=2,
        label='init traj'
    )
    ax.scatter(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        c='white',
        s=25,
        edgecolors='black',
        linewidths=0.5,
        zorder=4,
        label='init waypoints'
    )

    # ---- optimized trajectory ----
    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='opt traj'
    )
    ax.scatter(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        c='red',
        s=30,
        edgecolors='black',
        linewidths=0.5,
        zorder=5,
        label='opt waypoints'
    )

    ax.scatter(
        traj_opt_pts[0, 0],
        traj_opt_pts[0, 1],
        c='lime',
        s=80,
        zorder=6,
        label='start'
    )
    ax.scatter(
        traj_opt_pts[-1, 0],
        traj_opt_pts[-1, 1],
        c='red',
        s=80,
        zorder=6,
        label='goal'
    )

    draw_fov_wedges(
        ax,
        traj_opt_pts,
        traj_opt.yaws
    )

    ax.set_title("SDF only")
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.show()


