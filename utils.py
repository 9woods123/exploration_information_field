import math
import numpy as np
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.lines import Line2D
import time

mpl.rcParams['pdf.fonttype'] = 42     # TrueType
mpl.rcParams['ps.fonttype']  = 42



class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self, msg):
        t1 = time.perf_counter()
        dt_ms = (t1 - self.t0) * 1000.0
        print(f"[TIMER] {msg:35s}: {dt_ms:8.2f} ms")
        self.t0 = t1

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
    frame="traj_eif_sdf.pdf"
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
    
    obs_xy = []
    for idx, p in map2d.grid.items():
        if p > 0.9:
            obs_xy.append(map2d.grid_to_world(idx))

    obs_xy = np.array(obs_xy)

    plt.scatter(
        obs_xy[:,0], obs_xy[:,1],
        c='black', s=20, label='obstacles'
    )

    plt.tight_layout()

    # -----------------------------
    # save figure (IEEE-safe)
    # -----------------------------

    save_path = os.path.join(save_dir, frame)
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



def plot_eif_and_sdf(eif_table, sdf_field, Yaw_grid_2d, map2d, show_sdf=True,
    save_dir="results",
    frame="eif_sdf.pdf"):
                     
    """
    Visualize EIF field with optimal yaw, and (optionally) SDF field.

    Args:
        eif_table: contains xs, ys, I
        sdf_field: contains sdf
        Yaw_grid_2d: optimal yaw angle at each grid (same shape as I)
        map2d: occupancy map (with unknown + grid_to_world)
        show_sdf: True -> EIF + SDF (2 subplots)
                  False -> EIF only
    """

    xs = eif_table.xs
    ys = eif_table.ys
    I_grid = eif_table.I
    SDF = sdf_field.sdf

    X, Y = np.meshgrid(xs, ys, indexing='ij')

    # yaw vector field
    Ux = np.cos(Yaw_grid_2d)
    Uy = np.sin(Yaw_grid_2d)

    # unknown space
    unknown_xy = np.array(
        [map2d.grid_to_world(idx) for idx in map2d.unknown]
    )

    # -------------------------------------------------
    # Create canvas
    # -------------------------------------------------
    if show_sdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_eif, ax_sdf = axes
    else:
        fig, ax_eif = plt.subplots(1, 1, figsize=(7, 6))
        ax_sdf = None

    # =====================================================
    # EIF FIELD + YAW
    # =====================================================
    ax = ax_eif

    # Unknown space
    if len(unknown_xy) > 0:
        ax.scatter(
            unknown_xy[:, 0],
            unknown_xy[:, 1],
            s=5,
            c='lightgray',
            alpha=0.6,
            label='unknown space'
        )

    # EIF contour

    c1 = ax.contourf(
        X, Y, I_grid,
        levels=30,
        cmap='viridis'
    )


    cbar1 = fig.colorbar(c1, ax=ax, shrink=0.8)
    cbar1.set_label("Expected Information Gain")

    # Yaw field
    # ---- yaw vector field ----

    # ---- downsample factor ----
    step = 2   # 每4个格子取1个（你可以改成 3,5,6 试试）

    X_ds = X[::step, ::step]
    Y_ds = Y[::step, ::step]
    Ux_ds = Ux[::step, ::step]
    Uy_ds = Uy[::step, ::step]

    ax.quiver(
        X_ds, Y_ds, Ux_ds, Uy_ds,
        color='red',
        alpha=0.4,
        scale=35,
        width=0.004
    )
    # ax.quiver(
    #     X, Y, Ux, Uy,
    #     color='red',
    #     alpha=0.4,
    #     scale=60,
    #     width=0.0025
    # )

    # ---- legend arrow ----

    yaw_handle = Line2D(
        [0], [0],
        color='red',
        marker=r'$\rightarrow$',
        linestyle='None',
        markersize=12,
        label='optimal yaw'
    )

        # obstacles
    obs_xy = []
    for idx, p in map2d.grid.items():
        if p > 0.9:
            obs_xy.append(map2d.grid_to_world(idx))
    obs_xy = np.array(obs_xy)

    if len(obs_xy) > 0:
        ax.scatter(
            obs_xy[:, 0],
            obs_xy[:, 1],
            c='black',
            alpha=1,
            s=2,
            label='obstacles'
        )

    ax.set_title("Expected Information Field (with Optimal Yaw)")
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(yaw_handle)

    ax.legend(handles=handles, loc='upper right', fontsize=8)
    # =====================================================
    # SDF FIELD (Optional)
    # =====================================================
    if show_sdf:
        ax = ax_sdf

        c2 = ax.contourf(
            X, Y, SDF,
            levels=40,
            cmap='coolwarm'
        )
        cbar2 = fig.colorbar(c2, ax=ax, shrink=0.8)
        cbar2.set_label("Signed Distance Field")

        # zero level set = obstacle boundary
        # ax.contour(
        #     X, Y, SDF,
        #     levels=[0.0],
        #     colors='black',
        #     linewidths=2,
        #     label='obstacle boundary'
        # )

        # obstacles
        obs_xy = []
        for idx, p in map2d.grid.items():
            if p > 0.9:
                obs_xy.append(map2d.grid_to_world(idx))
        obs_xy = np.array(obs_xy)

        if len(obs_xy) > 0:
            ax.scatter(
                obs_xy[:, 0],
                obs_xy[:, 1],
                c='black',
                s=20,
                label='obstacles'
            )

        ax.set_title("Signed Distance Field")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)


    plt.tight_layout()

    # -----------------------------
    # save figure (IEEE-safe)
    # -----------------------------

    save_path = os.path.join(save_dir, frame)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Figure saved] {save_path}")
    
    plt.show()




def plot_traj_and_fieldmap(
    eif_table,
    sdf_field,
    map2d,
    traj0,
    traj_opt,
    traj_sdf_only_opt,
    save_dir="results",
    frame="traj_eif_sdf.pdf",
    show_sdf=True
):
    # -----------------------------
    # prepare fields
    # -----------------------------
    xs = eif_table.xs
    ys = eif_table.ys
    I_grid = eif_table.I
    SDF = sdf_field.sdf

    X, Y = np.meshgrid(xs, ys, indexing='ij')

    unknown_xy = np.array(
        [map2d.grid_to_world(idx) for idx in map2d.unknown]
    ) if len(map2d.unknown) > 0 else np.empty((0, 2))

    traj0_pts = traj0.waypoints
    traj_opt_pts = traj_opt.waypoints
    traj_sdf_pts = traj_sdf_only_opt.waypoints

    # -----------------------------
    # figure layout
    # -----------------------------
    if show_sdf:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax_eif, ax_sdf = axes
    else:
        fig, ax_eif = plt.subplots(1, 1, figsize=(7, 6))

    # =====================================================
    # LEFT / ONLY: EIF FIELD + trajectories
    # =====================================================
    ax = ax_eif

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

    # EIF field
    c1 = ax.contourf(
        X, Y, I_grid,
        levels=30,
        cmap='viridis'
    )
    fig.colorbar(c1, ax=ax, shrink=0.8, label="EIF")

    # init traj
    ax.plot(
        traj0_pts[:, 0],
        traj0_pts[:, 1],
        '--',
        color='pink',
        linewidth=1.8,
        label='init traj'
    )

    # EIF + SDF optimized
    ax.plot(
        traj_opt_pts[:, 0],
        traj_opt_pts[:, 1],
        '-r',
        linewidth=2.5,
        label='EIF + SDF opt'
    )

    # SDF only optimized
    ax.plot(
        traj_sdf_pts[:, 0],
        traj_sdf_pts[:, 1],
        '-c',
        linewidth=2.2,
        label='SDF only opt'
    )

    obs_xy = []
    for idx, p in map2d.grid.items():
        if p > 0.9:
            obs_xy.append(map2d.grid_to_world(idx))
    obs_xy = np.array(obs_xy)

    if len(obs_xy) > 0:
        ax.scatter(
            obs_xy[:, 0],
            obs_xy[:, 1],
            c='black',
            alpha=1,
            s=45,
            label='obstacles'
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

    # FOV wedges (best yaw)
    draw_fov_wedges(ax, traj_opt_pts, traj_opt.yaws)

    ax.set_title("EIF field + trajectories")
    ax.set_aspect('equal')
    ax.legend(
        loc='upper right',
        fontsize=6,
        framealpha=0.85
    )

    # =====================================================
    # RIGHT: SDF FIELD + trajectories (optional)
    # =====================================================
    if show_sdf:
        ax = ax_sdf

        c2 = ax.contourf(
            X, Y, SDF,
            levels=40,
            cmap='coolwarm'
        )
        fig.colorbar(c2, ax=ax, shrink=0.8, label="SDF")

        # obstacle boundary (SDF = 0)

        # init traj
        ax.plot(
            traj0_pts[:, 0],
            traj0_pts[:, 1],
            '--',
            color='pink',
            linewidth=1.8,
            label='init traj'
        )

        # EIF + SDF optimized
        ax.plot(
            traj_opt_pts[:, 0],
            traj_opt_pts[:, 1],
            '-r',
            linewidth=2.5,
            label='EIF + SDF opt'
        )

        # SDF only optimized
        ax.plot(
            traj_sdf_pts[:, 0],
            traj_sdf_pts[:, 1],
            '-c',
            linewidth=2.2,
            label='SDF only opt'
        )
        # obstacle points (solid)
        obs_xy = []
        for idx, p in map2d.grid.items():
            if p > 0.9:
                obs_xy.append(map2d.grid_to_world(idx))

        if len(obs_xy) > 0:
            obs_xy = np.array(obs_xy)
            ax.scatter(
                obs_xy[:, 0],
                obs_xy[:, 1],
                c='black',
                s=20,
                label='obstacles'
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

        draw_fov_wedges(ax, traj_opt_pts, traj_opt.yaws)


        ax.set_title("SDF field + trajectories")
        ax.set_aspect('equal')
        ax.legend(
            loc='upper right',
            fontsize=6,
            framealpha=0.85
        )

    # -----------------------------
    # save figure
    # -----------------------------
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, frame)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Figure saved] {save_path}")

    plt.show()
