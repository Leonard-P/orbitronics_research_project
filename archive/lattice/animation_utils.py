"""
Animation utilities for rendering DynamicsFrameRecorderGPU outputs.

This module provides:
- bond_current_for_pair: retrieve the time series for a specific edge (k,l)
- save_current_density_animation: save an animation of onsite densities and bond currents
- Optional overlays:
    - Flow-direction arrows using a single Quiver artist (efficient)
    - Curl indicators (scalar per-cell) using a single scatter artist (efficient)

All updates are vectorized (scatter colors + Quiver.set_UVC) for efficiency.
"""
from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import animation
import warnings
from matplotlib.patches import Arc, RegularPolygon
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

# For geometry-specific curl offset
try:
    from .lattice_geometry import HexagonalLatticeGeometry
except Exception:  # pragma: no cover - optional import for type/instance checks
    HexagonalLatticeGeometry = object  # fallback to avoid NameError


def bond_current_for_pair(
    animation_values: dict,
    nn_rows: np.ndarray,
    nn_cols: np.ndarray,
    k: int,
    l: int,
) -> np.ndarray:
    """Return J_{k,l}(t) from DynamicsFrameRecorderGPU results.

    If the stored edge list contains (k,l) directly, returns that series.
    If it contains (l,k) instead, returns the negated series.
    Raises KeyError if neither direction is present.
    """
    rows = np.asarray(nn_rows).ravel()
    cols = np.asarray(nn_cols).ravel()
    J = animation_values["bond_currents"]  # shape (F, E)

    idx = np.where((rows == k) & (cols == l))[0]
    if idx.size:
        return J[:, idx[0]]

    idx = np.where((rows == l) & (cols == k))[0]
    if idx.size:
        return -J[:, idx[0]]

    raise KeyError(f"Edge ({k},{l}) not found in recorded edges.")


def _build_geometry_segments(
    nn_rows: np.ndarray,
    nn_cols: np.ndarray,
    site_to_position,
) -> np.ndarray:
    """Build line segments array given an edge list and a position function.

    Returns array of shape (E, 2, 2): [ [ (xk, yk), (xl, yl) ], ... ].
    """
    rows = np.asarray(nn_rows).ravel()
    cols = np.asarray(nn_cols).ravel()
    E = rows.shape[0]
    segs = np.empty((E, 2, 2), dtype=float)
    for e in range(E):
        xk, yk = site_to_position(int(rows[e]))
        xl, yl = site_to_position(int(cols[e]))
        segs[e, 0, 0] = xk
        segs[e, 0, 1] = yk
        segs[e, 1, 0] = xl
        segs[e, 1, 1] = yl
    return segs


def _site_coordinates(N: int, site_to_position) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.empty(N, dtype=float)
    ys = np.empty(N, dtype=float)
    for i in range(N):
        x, y = site_to_position(i)
        xs[i] = x
        ys[i] = y
    return xs, ys


def save_current_density_animation(
    geometry,
    animation_values: dict,
    nn_rows: np.ndarray,
    nn_cols: np.ndarray,
    out_path: str,
    fps: int = 20,
    dpi: int = 150,
    density_cmap: str = "Greys",
    current_cmap: str = "bwr",  # retained for legend color choice if needed
    density_vmin: Optional[float] = 0.0,
    density_vmax: Optional[float] = 1.0,
    current_max: Optional[float] = None,
    # Site circles size
    site_marker_size: float = 320.0,
    # Flow-direction arrows (efficient: one Quiver artist)
    show_flow_arrows: bool = True,
    arrows_per_edge: int = 3,
    arrow_scale: float = 0.55,
    arrow_width: float = 0.04,
    arrow_color: str = "black",
    # Curl indicators (efficient: one scatter at cell centers)
    show_curl_indicators: bool = True,
    curl_cmap: str = "RdBu",
    curl_vmax: Optional[float] = None,
    curl_marker_size: float = 180.0,
    # Optional curl direction circular arrows (pre-created patches, toggle per frame)
    show_curl_direction_arrows: bool = True,
    curl_arrow_radius: float = 0.6,
    curl_arrow_lw: float = 1.5,
    curl_arrow_positive_color: str = "red",
    curl_arrow_negative_color: str = "blue",
    curl_arrow_threshold: float = 0.01,
    # Optional per-frame text at top-left
    frame_texts: Optional[List[str]] = None,
) -> None:
    """Save an animation visualizing onsite densities and bond currents over frames.

    - geometry: LatticeGeometry (needs site_to_position, dimensions)
    - animation_values: dict returned by DynamicsFrameRecorderGPU.finalize()
    - nn_rows/nn_cols: the same arrays used to record bond currents
    - out_path: output file (e.g., mp4 or gif depending on writer)
    - current_max: normalization for currents; if None, derived from data
    - show_flow_arrows: arrows along each edge, oriented by the sign of current
    - show_curl_indicators: per-cell scalar curl (uses animation_values['orbital_curl'] if present)
    - frame_texts: if provided, uses these strings per frame as the title text
    """
    densities = animation_values["densities"]  # (F, N)
    bond_currents = animation_values["bond_currents"]  # (F, E)

    F, N = densities.shape
    _, E = bond_currents.shape
    if F == 0:
        raise ValueError("No frames to animate.")

    # Coordinates
    xs, ys = _site_coordinates(N, geometry.site_to_position)
    segments = _build_geometry_segments(nn_rows, nn_cols, geometry.site_to_position)

    # Normalizations
    if current_max is None:
        # robust peak current scale; fallback to 1.0 if zeros
        current_max = float(np.max(np.abs(bond_currents)))
        if current_max == 0:
            current_max = 1.0

    # Figure
    Lx, Ly = geometry.dimensions
    fig_width = max(1, 0.8 * int(Lx))
    fig_height = max(1, 1 * int(Ly))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # Tighten padding and reserve right space for legend/colorbars
    fig.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.06)

    # Artists
    # Densities as scatter; set_array updates per frame
    dens0 = densities[0]
    sc = ax.scatter(
        xs,
        ys,
        c=dens0,
        cmap=density_cmap,
        vmin=density_vmin,
        vmax=density_vmax,
        s=site_marker_size,
        edgecolor="black",
        linewidths=0.6,
        zorder=2,
    )

    # Optional: flow-direction arrows via Quiver (single artist, efficient)
    quiv = None
    dirx = diry = None
    if show_flow_arrows and E > 0 and arrows_per_edge > 0:
        # Build arrow centers along each segment and unit direction per edge
        P0 = segments[:, 0, :]
        P1 = segments[:, 1, :]
        dP = P1 - P0
        lengths = np.linalg.norm(dP, axis=1)
        safe_lengths = np.where(lengths == 0, 1.0, lengths)
        dirs = dP / safe_lengths[:, None]  # (E,2)

        # Fractional positions along each edge for arrow markers
        fracs = (np.arange(1, arrows_per_edge + 1) / (arrows_per_edge + 1)).astype(float)
        Px = (P0[:, 0:1] + fracs * dP[:, 0:1]).reshape(-1)
        Py = (P0[:, 1:2] + fracs * dP[:, 1:2]).reshape(-1)
        # Repeat per-edge unit directions K times
        dirx = np.repeat(dirs[:, 0], arrows_per_edge)
        diry = np.repeat(dirs[:, 1], arrows_per_edge)
        # Initial U,V from first frame sign and magnitude (length ~ |J|/current_max)
        J0 = bond_currents[0]
        s0 = np.sign(J0)
        L0 = arrow_scale * np.repeat(np.abs(J0) / current_max, arrows_per_edge)
        U0 = L0 * np.repeat(s0, arrows_per_edge) * dirx
        V0 = L0 * np.repeat(s0, arrows_per_edge) * diry

        quiv = ax.quiver(
            Px,
            Py,
            U0,
            V0,
            units="xy",
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="middle",
            color=arrow_color,
            width=arrow_width,
            headwidth=7,
            headlength=8,
            headaxislength=7,
            linewidth=0,
            zorder=1,
        )

    # Optional: curl indicators at cell centers (single scatter)
    curl_sc = None
    curl_sites = None
    curl_vals0 = None
    curl_all = None
    # For circular arrows
    curl_ccw_arcs = []
    curl_ccw_heads = []
    curl_cw_arcs = []
    curl_cw_heads = []
    if show_curl_indicators:
        if "orbital_curl" in animation_values:
            curl_all = animation_values["orbital_curl"]  # (F, C)
            curl_vals0 = np.asarray(curl_all[0])
            try:
                curl_sites = list(geometry.get_curl_sites())
            except Exception as e:
                warnings.warn(f"Could not retrieve curl sites from geometry: {e}. Disabling curl indicators.")
                show_curl_indicators = False
        else:
            warnings.warn(
                "animation_values lacks 'orbital_curl'; curl indicators disabled. "
                "Record curl in DynamicsFrameRecorderGPU for efficient overlays."
            )
            show_curl_indicators = False

        if show_curl_indicators and curl_vals0 is not None:
            # Compute anchor positions for curl glyphs (with offsets)
            cx = np.empty(len(curl_sites), dtype=float)
            cy = np.empty(len(curl_sites), dtype=float)
            for i, s in enumerate(curl_sites):
                x, y = geometry.site_to_position(int(s))
                x += 0.5
                y += 0.5
                if isinstance(geometry, HexagonalLatticeGeometry):
                    x += np.sqrt(3) / 2 - 0.5
                cx[i] = x
                cy[i] = y

            if curl_vmax is None:
                curl_vmax = float(np.max(np.abs(curl_all))) if np.size(curl_all) else 1.0
                if curl_vmax == 0:
                    curl_vmax = 1.0

            curl_sc = ax.scatter(
                cx,
                cy,
                c=curl_vals0,
                cmap=curl_cmap,
                vmin=-curl_vmax,
                vmax=curl_vmax,
                s=curl_marker_size,
                edgecolor="none",
                zorder=3,
                alpha=1.0,
            )

            # Optional: circular direction arrows per cell (pre-create, toggle visibility)
            if show_curl_direction_arrows:
                angle_ = 125
                theta2_ = 310
                for i in range(len(curl_sites)):
                    x = cx[i]
                    y = cy[i]
                    arc_ccw = Arc(
                        (x, y),
                        curl_arrow_radius,
                        curl_arrow_radius,
                        angle=angle_,
                        theta1=0,
                        theta2=theta2_,
                        capstyle="round",
                        linestyle="-",
                        lw=curl_arrow_lw,
                        color=curl_arrow_positive_color,
                        zorder=4,
                        alpha=1.0,
                    )
                    endX_ccw = x + (curl_arrow_radius / 2.0) * np.cos(np.radians(theta2_ + angle_))
                    endY_ccw = y + (curl_arrow_radius / 2.0) * np.sin(np.radians(theta2_ + angle_))
                    orient_ccw = np.radians(angle_ + theta2_)
                    head_ccw = RegularPolygon(
                        (endX_ccw, endY_ccw),
                        3,
                        radius=curl_arrow_radius / 7.0,
                        orientation=orient_ccw,
                        color=curl_arrow_positive_color,
                        zorder=5,
                        alpha=1.0,
                    )
                    ax.add_patch(arc_ccw)
                    ax.add_patch(head_ccw)
                    curl_ccw_arcs.append(arc_ccw)
                    curl_ccw_heads.append(head_ccw)

                    arc_cw = Arc(
                        (x, y),
                        curl_arrow_radius,
                        curl_arrow_radius,
                        angle=angle_,
                        theta1=0,
                        theta2=theta2_,
                        capstyle="round",
                        linestyle="-",
                        lw=curl_arrow_lw,
                        color=curl_arrow_negative_color,
                        zorder=4,
                        alpha=1.0,
                    )
                    endX_cw = x + (curl_arrow_radius / 2.0) * np.cos(np.radians(angle_))
                    endY_cw = y + (curl_arrow_radius / 2.0) * np.sin(np.radians(angle_))
                    orient_cw = np.radians(angle_) + np.pi
                    head_cw = RegularPolygon(
                        (endX_cw, endY_cw),
                        3,
                        radius=curl_arrow_radius / 7.0,
                        orientation=orient_cw,
                        color=curl_arrow_negative_color,
                        zorder=5,
                        alpha=1.0,
                    )
                    ax.add_patch(arc_cw)
                    ax.add_patch(head_cw)
                    curl_cw_arcs.append(arc_cw)
                    curl_cw_heads.append(head_cw)

                # Initialize visibility according to first frame
                norm0 = curl_vals0 / (curl_vmax if curl_vmax else 1.0)
                for i, v in enumerate(norm0):
                    show_cw = (v >= curl_arrow_threshold)
                    show_ccw = (v <= -curl_arrow_threshold)
                    curl_ccw_arcs[i].set_visible(show_ccw)
                    curl_ccw_heads[i].set_visible(show_ccw)
                    curl_cw_arcs[i].set_visible(show_cw)
                    curl_cw_heads[i].set_visible(show_cw)

    # Title / per-frame text at top-left
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    if frame_texts and len(frame_texts) > 0:
        title.set_text(frame_texts[0])
    else:
        title.set_text(f"frame 1/{F}")

    # --- Legend and colorbars in a right-side vertical column ---
    handles: List[mlines.Line2D] = []
    labels: List[str] = []

    # Nearest-neighbor current (arrow-like handle)
    handles.append(
        mlines.Line2D([], [], color="black", linestyle="-", marker=">", markersize=8, mfc="black", mec="black", lw=1.8)
    )
    labels.append("Bond Current")

    # Site occupation (circle patch with border)
    occ_color = cm.get_cmap(density_cmap)(0.6)
    handles.append(
        mlines.Line2D([], [], color=occ_color, marker="o", linestyle="None", markersize=11, markeredgecolor="black", mew=1.0)
    )
    labels.append("Site Occupation $\\langle \\hat n_i\\rangle $")

    # Plaquette OAM (curl) density (circle, edge same as face)
    oam_color = cm.get_cmap(curl_cmap)(0.75)
    handles.append(
        mlines.Line2D([], [], color=oam_color, marker="o", linestyle="None", markersize=11, markeredgecolor=oam_color)
    )
    labels.append("Plaquette OAM")

    # Place legend at top-right outside the axes
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),  # top of the right-side column
        bbox_transform=ax.transAxes,
        fontsize="medium",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        labelspacing=1.0,
        handletextpad=0.8,
        handlelength=1.8,
        borderpad=0.5,
    )

    # Draw to compute legend bbox for placing stacked colorbars beneath it
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())

    # Standard proportions for colorbars
    cbar_w = 0.015   # slim vertical bar
    cbar_h = 2 / geometry.Ly  # ~22% of figure height
    cbar_spacing = 0.02  # vertical spacing between bars

    # Center colorbars under the legend box horizontally
    cbar_x = legend_bbox.x0 + (legend_bbox.width - cbar_w) / 2.0

    # First (top) colorbar: site occupation
    occ_norm = Normalize(
        vmin=density_vmin if density_vmin is not None else np.nanmin(densities),
        vmax=density_vmax if density_vmax is not None else np.nanmax(densities),
    )
    occ_sm = cm.ScalarMappable(norm=occ_norm, cmap=plt.get_cmap(density_cmap))
    occ_sm.set_array([])
    # Position directly below legend with a small gap
    occ_y = max(0.06, legend_bbox.y0 - 0.02 - cbar_h)
    cax_occ = fig.add_axes([cbar_x, occ_y, cbar_w, cbar_h])
    cb_occ = fig.colorbar(occ_sm, cax=cax_occ, orientation="vertical")
    cb_occ.set_label("Site Occupation $\\langle \\hat n_i\\rangle $", size="small")
    cb_occ.ax.tick_params(labelsize="small")

    # Second (bottom) colorbar: OAM/curl if available

    if show_curl_indicators and curl_sc is not None:
        oam_norm = Normalize(vmin=-curl_vmax, vmax=curl_vmax)
        oam_sm = cm.ScalarMappable(norm=oam_norm, cmap=plt.get_cmap(curl_cmap))
        oam_sm.set_array([])
        oam_y = max(0.06, occ_y - cbar_spacing - cbar_h)
        cax_oam = fig.add_axes([cbar_x, oam_y, cbar_w, cbar_h])
        cb_oam = fig.colorbar(oam_sm, cax=cax_oam, orientation="vertical")
        cb_oam.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        cb_oam.formatter.set_powerlimits((-2, 2))
        cb_oam.update_ticks()
        cb_oam.set_label("Plaquette OAM ($\\hbar$)", size="small")
        cb_oam.ax.tick_params(labelsize="small")

    def update(frame: int):
        d = densities[frame]
        sc.set_array(d)
        # Update flow-direction arrows
        artists = [sc, title]
        if quiv is not None:
            J = bond_currents[frame]
            sgn = np.sign(J)
            L = arrow_scale * np.repeat(np.abs(J) / current_max, arrows_per_edge)
            U = L * np.repeat(sgn, arrows_per_edge) * dirx
            V = L * np.repeat(sgn, arrows_per_edge) * diry
            quiv.set_UVC(U, V)
            artists.append(quiv)
        # Update curl indicators
        if curl_sc is not None and "orbital_curl" in animation_values:
            vals = np.asarray(animation_values["orbital_curl"][frame])
            curl_sc.set_array(vals)
            artists.append(curl_sc)
            if show_curl_direction_arrows and curl_ccw_arcs and curl_cw_arcs:
                denom = (curl_vmax if curl_vmax else (np.max(np.abs(vals)) or 1.0))
                normv = vals / denom
                for i, v in enumerate(normv):
                    show_cw = (v >= curl_arrow_threshold)
                    show_ccw = (v <= -curl_arrow_threshold)
                    curl_ccw_arcs[i].set_visible(show_ccw)
                    curl_ccw_heads[i].set_visible(show_ccw)
                    curl_cw_arcs[i].set_visible(show_cw)
                    curl_cw_heads[i].set_visible(show_cw)
        # Update frame text
        if frame_texts and frame < len(frame_texts):
            title.set_text(frame_texts[frame])
        else:
            title.set_text(f"frame {frame+1}/{F}")
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=F, interval=1000 // max(1, fps), blit=False)

    # Try ffmpeg, else PillowWriter
    try:
        anim.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi)
    except Exception:
        from matplotlib.animation import PillowWriter
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)

    plt.close(fig)
