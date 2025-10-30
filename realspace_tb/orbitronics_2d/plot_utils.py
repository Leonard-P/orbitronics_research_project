"""
Animation utilities for plotting and rendering animations of 2D Lattice geometries.

"""

from typing import cast, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import animation
from matplotlib.patches import Arc, RegularPolygon
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
from traitlets import List

from realspace_tb.orbitronics_2d.honeycomb_geometry import HoneycombLatticeGeometry

from .lattice_2d_geometry import Lattice2DGeometry
from .. import backend as B
from .observables import LatticeFrameObservable


def _build_geometry_segments(geometry: Lattice2DGeometry) -> np.ndarray:
    """Build line segments array for nearest-neighbor bonds from the geometry's edge list and position function.
    Parameters:
        geometry: Lattice2DGeometry with nearest_neighbors and index_to_position defined.

    Returns:
        array of shape (E, 2, 2): [ [ (xk, yk), (xl, yl) ], ... ].
    """
    rows = geometry.nearest_neighbors[:, 0]
    cols = geometry.nearest_neighbors[:, 1]
    E = rows.shape[0]
    segs = np.empty((E, 2, 2), dtype=float)
    for e in range(E):
        xk, yk = geometry.index_to_position(int(rows[e]))
        xl, yl = geometry.index_to_position(int(cols[e]))
        segs[e, 0, 0] = xk
        segs[e, 0, 1] = yk
        segs[e, 1, 0] = xl
        segs[e, 1, 1] = yl
    return segs


def _site_coordinates(geometry: Lattice2DGeometry) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "This function has been deprecated. Use geometry.site_positions instead."
    )
    N = geometry.Lx * geometry.Ly
    xs = np.empty(N, dtype=float)
    ys = np.empty(N, dtype=float)
    for i in range(N):
        x, y = geometry.index_to_position(i)
        xs[i] = x
        ys[i] = y
    return xs, ys


def _create_scene(
    lattice_frame_obs: LatticeFrameObservable,
    geometry: Lattice2DGeometry,
    density_cmap: str = "Greys",
    density_vmin: float = 0.0,
    density_vmax: float = 1.0,
    current_max: float | None = None,
    # Site circles size
    site_marker_size: float = 320.0,
    # Flow-direction arrows
    show_flow_arrows: bool = True,
    arrows_per_edge: int = 3,
    arrow_scale: float = 0.55,
    arrow_width: float = 0.04,
    arrow_color: str = "black",
    # OAM indicators
    show_oam_indicators: bool = True,
    oam_cmap: str = "RdBu",
    oam_vmax: float | None = None,
    oam_marker_size: float = 180.0,
    # curl direction circular arrows
    show_oam_direction_arrows: bool = True,
    oam_arrow_radius: float = 0.6,
    oam_arrow_lw: float = 1.5,
    oam_arrow_positive_color: str = "red",
    oam_arrow_negative_color: str = "blue",
    # Hide arrows for small curl values
    oam_arrow_threshold: float | None = None,
    # Optional per-frame text at top-left
    frame_texts: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    """Builds the static scene (figure, artists, legend, colorbars) and returns a context dict for updating per-frame."""
    animation_values = cast(dict[str, B.FCPUArray], lattice_frame_obs.values)
    densities = animation_values["densities"]  # (F, N)
    bond_currents = animation_values["currents"]  # (F, E)

    F, N = densities.shape
    _, E = bond_currents.shape

    # Coordinates
    xs = geometry.site_positions[:, 0]
    ys = geometry.site_positions[:, 1]
    segments = _build_geometry_segments(geometry)

    # Normalizations
    if current_max is None:
        current_max = float(np.max(np.abs(bond_currents)))
        if current_max == 0:
            current_max = 1.0

    # Figure
    Lx, Ly = geometry.Lx, geometry.Ly
    fig_width = max(1, 0.7 * int(Lx)) + 2.0  # reserve right space for legend/colorbars
    fig_height = max(1, 1 * int(Ly))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0.01, right=1 - 1 / Lx, top=0.99, bottom=0.01)

    # Densities as scatter
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

    # Flow-direction arrows via Quiver
    quiv = None
    dirx = diry = None
    if show_flow_arrows and E > 0 and arrows_per_edge > 0:
        P0 = segments[:, 0, :]
        P1 = segments[:, 1, :]
        dP = P1 - P0
        lengths = np.linalg.norm(dP, axis=1)
        safe_lengths = np.where(lengths == 0, 1.0, lengths)
        dirs = dP / safe_lengths[:, None]  # (E,2)

        fracs = (np.arange(1, arrows_per_edge + 1) / (arrows_per_edge + 1)).astype(
            float
        )
        Px = (P0[:, 0:1] + fracs * dP[:, 0:1]).reshape(-1)
        Py = (P0[:, 1:2] + fracs * dP[:, 1:2]).reshape(-1)
        dirx = np.repeat(dirs[:, 0], arrows_per_edge)
        diry = np.repeat(dirs[:, 1], arrows_per_edge)
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

    # OAM indicators and optional circular arrows
    curl_sc = None
    curl_all = None
    curl_sites = None
    curl_ccw_arcs: list[Arc] = []
    curl_ccw_heads: list[RegularPolygon] = []
    curl_cw_arcs: list[Arc] = []
    curl_cw_heads: list[RegularPolygon] = []
    oam_vmax_f: float = 1.0
    oam_arrow_threshold_f: float = 0.01

    if show_oam_indicators:
        curl_all = animation_values["plaquette_oam"]  # (F, C)
        curl_vals0 = np.asarray(curl_all[0])
        curl_sites = lattice_frame_obs.plaquette_anchor_indices
        curl_pos = geometry.site_positions[curl_sites.astype(int)]

        if not isinstance(geometry, HoneycombLatticeGeometry):
            raise NotImplementedError(
                "OAM indicators are only implemented for HoneycombLatticeGeometry. Plaquette Center Offsets vary for other geometries."
            )

        cx = curl_pos[:, 0] + np.sqrt(3) / 2
        cy = curl_pos[:, 1] + 0.5

        if oam_vmax is None:
            oam_vmax_f = float(np.max(np.abs(curl_all))) if np.size(curl_all) else 1.0
            if oam_vmax_f == 0:
                oam_vmax_f = 1.0
        else:
            oam_vmax_f = float(oam_vmax)

        oam_arrow_threshold_f = (
            (0.01 * oam_vmax_f)
            if oam_arrow_threshold is None
            else float(oam_arrow_threshold)
        )

        curl_sc = ax.scatter(
            cx,
            cy,
            c=curl_vals0,
            cmap=oam_cmap,
            vmin=-oam_vmax_f,
            vmax=oam_vmax_f,
            s=oam_marker_size,
            edgecolor="none",
            zorder=3,
            alpha=1.0,
        )

        if show_oam_direction_arrows:
            angle_ = 125
            theta2_ = 310
            for i in range(len(curl_sites)):
                x = cx[i]
                y = cy[i]
                # CCW
                arc_ccw = Arc(
                    (x, y),
                    oam_arrow_radius,
                    oam_arrow_radius,
                    angle=angle_,
                    theta1=0,
                    theta2=theta2_,
                    capstyle="round",
                    linestyle="-",
                    lw=oam_arrow_lw,
                    color=oam_arrow_positive_color,
                    zorder=4,
                    alpha=1.0,
                )
                endX_ccw = x + (oam_arrow_radius / 2.0) * np.cos(
                    np.radians(theta2_ + angle_)
                )
                endY_ccw = y + (oam_arrow_radius / 2.0) * np.sin(
                    np.radians(theta2_ + angle_)
                )
                orient_ccw = np.radians(angle_ + theta2_)
                head_ccw = RegularPolygon(
                    (endX_ccw, endY_ccw),
                    3,
                    radius=oam_arrow_radius / 7.0,
                    orientation=orient_ccw,
                    color=oam_arrow_positive_color,
                    zorder=5,
                    alpha=1.0,
                )
                ax.add_patch(arc_ccw)
                ax.add_patch(head_ccw)
                curl_ccw_arcs.append(arc_ccw)
                curl_ccw_heads.append(head_ccw)
                # CW
                arc_cw = Arc(
                    (x, y),
                    oam_arrow_radius,
                    oam_arrow_radius,
                    angle=angle_,
                    theta1=0,
                    theta2=theta2_,
                    capstyle="round",
                    linestyle="-",
                    lw=oam_arrow_lw,
                    color=oam_arrow_negative_color,
                    zorder=4,
                    alpha=1.0,
                )
                endX_cw = x + (oam_arrow_radius / 2.0) * np.cos(np.radians(angle_))
                endY_cw = y + (oam_arrow_radius / 2.0) * np.sin(np.radians(angle_))
                orient_cw = np.radians(angle_) + np.pi
                head_cw = RegularPolygon(
                    (endX_cw, endY_cw),
                    3,
                    radius=oam_arrow_radius / 7.0,
                    orientation=orient_cw,
                    color=oam_arrow_negative_color,
                    zorder=5,
                    alpha=1.0,
                )
                ax.add_patch(arc_cw)
                ax.add_patch(head_cw)
                curl_cw_arcs.append(arc_cw)
                curl_cw_heads.append(head_cw)

            # Initialize visibility from frame 0
            norm0 = curl_vals0 / (oam_vmax_f if oam_vmax_f else 1.0)
            for i, v in enumerate(norm0):
                show_cw = v >= oam_arrow_threshold_f
                show_ccw = v <= -oam_arrow_threshold_f
                curl_ccw_arcs[i].set_visible(show_ccw)
                curl_ccw_heads[i].set_visible(show_ccw)
                curl_cw_arcs[i].set_visible(show_cw)
                curl_cw_heads[i].set_visible(show_cw)

    # Title / per-frame text
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    if frame_texts and len(frame_texts) > 0:
        title.set_text(frame_texts[0])
    else:
        title.set_text(f"frame 1/{F}")

    # Legend and colorbars
    handles: list[mlines.Line2D] = []
    labels: list[str] = []
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="-",
            marker=">",
            markersize=8,
            mfc="black",
            mec="black",
            lw=1.8,
        )
    )
    labels.append("Bond Current")
    occ_color = cm.get_cmap(density_cmap)(0.6)
    handles.append(
        mlines.Line2D(
            [],
            [],
            color=occ_color,
            marker="o",
            linestyle="None",
            markersize=11,
            markeredgecolor="black",
            mew=1.0,
        )
    )
    labels.append("Site Occupation $\\langle \\hat n_i\\rangle $")
    oam_color = cm.get_cmap(oam_cmap)(0.75)
    handles.append(
        mlines.Line2D(
            [],
            [],
            color=oam_color,
            marker="o",
            linestyle="None",
            markersize=11,
            markeredgecolor=oam_color,
        )
    )
    labels.append("Plaquette OAM")

    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
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

    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    cbar_w = 0.3 / geometry.Lx
    cbar_h = 2 / geometry.Ly
    cbar_spacing = 0.02
    cbar_x = legend_bbox.x0 + (legend_bbox.width - cbar_w) / 2.0

    # Occupation colorbar
    occ_norm = Normalize(
        vmin=density_vmin if density_vmin is not None else np.nanmin(densities),
        vmax=density_vmax if density_vmax is not None else np.nanmax(densities),
    )
    occ_sm = cm.ScalarMappable(norm=occ_norm, cmap=plt.get_cmap(density_cmap))
    occ_sm.set_array([])
    occ_y = max(0.06, legend_bbox.y0 - 0.02 - cbar_h)
    cax_occ = fig.add_axes([cbar_x, occ_y, cbar_w, cbar_h])
    cb_occ = fig.colorbar(occ_sm, cax=cax_occ, orientation="vertical")
    cb_occ.set_label("Site Occupation", size="small")
    cb_occ.ax.tick_params(labelsize="small")

    # OAM colorbar
    if show_oam_indicators and curl_sc is not None:
        oam_norm = Normalize(vmin=-oam_vmax_f, vmax=oam_vmax_f)
        oam_sm = cm.ScalarMappable(norm=oam_norm, cmap=plt.get_cmap(oam_cmap))
        oam_sm.set_array([])
        oam_y = max(0.06, occ_y - cbar_spacing - cbar_h)
        cax_oam = fig.add_axes([cbar_x, oam_y, cbar_w, cbar_h])
        cb_oam = fig.colorbar(oam_sm, cax=cax_oam, orientation="vertical")
        cb_oam.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        cb_oam.formatter.set_powerlimits((-2, 2))
        cb_oam.update_ticks()
        cb_oam.set_label("Plaquette OAM ($\\hbar$)", size="small")
        cb_oam.ax.tick_params(labelsize="small")

    ctx: dict[str, Any] = {
        "F": F,
        "densities": densities,
        "bond_currents": bond_currents,
        "arrows_per_edge": arrows_per_edge,
        "arrow_scale": arrow_scale,
        "current_max": current_max,
        "dirx": dirx,
        "diry": diry,
        "sc": sc,
        "quiv": quiv,
        "title": title,
        "frame_texts": frame_texts,
        # OAM
        "curl_all": curl_all,
        "curl_sc": curl_sc,
        "show_oam_direction_arrows": show_oam_direction_arrows,
        "oam_vmax_f": oam_vmax_f,
        "oam_arrow_threshold_f": oam_arrow_threshold_f,
        "curl_ccw_arcs": curl_ccw_arcs,
        "curl_ccw_heads": curl_ccw_heads,
        "curl_cw_arcs": curl_cw_arcs,
        "curl_cw_heads": curl_cw_heads,
    }
    return fig, ax, ctx


def _update_scene(ctx: dict[str, Any], frame: int) -> tuple[plt.Artist, ...]:
    densities = ctx["densities"]
    bond_currents = ctx["bond_currents"]
    arrows_per_edge = ctx["arrows_per_edge"]
    arrow_scale = ctx["arrow_scale"]
    current_max = ctx["current_max"]
    dirx = ctx["dirx"]
    diry = ctx["diry"]
    sc = ctx["sc"]
    quiv = ctx["quiv"]
    title = ctx["title"]
    frame_texts = ctx["frame_texts"]

    artists: list[plt.Artist] = [sc, title]

    # Densities
    d = densities[frame]
    sc.set_array(d)

    # Currents arrows
    if (
        quiv is not None
        and dirx is not None
        and diry is not None
        and arrows_per_edge > 0
    ):
        J = bond_currents[frame]
        sgn = np.sign(J)
        L = arrow_scale * np.repeat(np.abs(J) / current_max, arrows_per_edge)
        U = L * np.repeat(sgn, arrows_per_edge) * dirx
        V = L * np.repeat(sgn, arrows_per_edge) * diry
        quiv.set_UVC(U, V)
        artists.append(quiv)

    # OAM indicators
    curl_sc = ctx["curl_sc"]
    curl_all = ctx["curl_all"]
    if curl_sc is not None and curl_all is not None:
        vals = np.asarray(curl_all[frame])
        curl_sc.set_array(vals)
        artists.append(curl_sc)

        if ctx["show_oam_direction_arrows"]:
            oam_vmax_f = ctx["oam_vmax_f"]
            oam_arrow_threshold_f = ctx["oam_arrow_threshold_f"]
            denom = oam_vmax_f if oam_vmax_f else (np.max(np.abs(vals)) or 1.0)
            normv = vals / denom
            for i, v in enumerate(normv):
                show_cw = v >= oam_arrow_threshold_f
                show_ccw = v <= -oam_arrow_threshold_f
                ctx["curl_ccw_arcs"][i].set_visible(show_ccw)
                ctx["curl_ccw_heads"][i].set_visible(show_ccw)
                ctx["curl_cw_arcs"][i].set_visible(show_cw)
                ctx["curl_cw_heads"][i].set_visible(show_cw)

    # Title text
    F = ctx["F"]
    if frame_texts and frame < len(frame_texts):
        title.set_text(frame_texts[frame])
    else:
        title.set_text(f"frame {frame+1}/{F}")

    return tuple(artists)


def save_simulation_animation(
    lattice_frame_obs: LatticeFrameObservable,
    geometry: Lattice2DGeometry,
    out_path: str,
    fps: int = 20,
    dpi: int = 150,
    density_cmap: str = "Greys",
    density_vmin: float = 0.0,
    density_vmax: float = 1.0,
    current_max: float | None = None,
    # Site circles size
    site_marker_size: float = 320.0,
    # Flow-direction arrows
    show_flow_arrows: bool = True,
    arrows_per_edge: int = 3,
    arrow_scale: float = 0.55,
    arrow_width: float = 0.04,
    arrow_color: str = "black",
    # OAM indicators
    show_oam_indicators: bool = True,
    oam_cmap: str = "RdBu",
    oam_vmax: float | None = None,
    oam_marker_size: float = 180.0,
    # curl direction circular arrows
    show_oam_direction_arrows: bool = True,
    oam_arrow_radius: float = 0.6,
    oam_arrow_lw: float = 1.5,
    oam_arrow_positive_color: str = "red",
    oam_arrow_negative_color: str = "blue",
    # Hide arrows for small curl values
    oam_arrow_threshold: float | None = None,
    # Optional per-frame text at top-left
    frame_texts: list[str] | None = None,
) -> None:
    """Save an animation visualizing onsite densities and bond currents over frames.

    Parameters:
        lattice_frame_obs: LatticeFrameObservable that recorded 'densities', 'currents', 'plaquette_oam' during the simulation
        geometry: Lattice2DGeometry providing site positions and nearest neighbor pairs
        out_path: output file (e.g., mp4 or gif)
        fps: frames per second in output animation
        dpi: resolution of output animation
        density_cmap: colormap for site densities
        density_vmin: min value for density colormap
        density_vmax: max value for density colormap
        current_max: max value for current colormap; if None, derived from data
        site_marker_size: size of site occupation circles
        show_flow_arrows: whether to show flow-direction arrows along bonds that indicate the current direction
        arrows_per_edge: number of arrows to draw along each bond
        arrow_scale: scaling factor for arrow lengths (w.r.t. current magnitude)
        arrow_width: width of arrows
        arrow_color: color of arrows
        show_oam_indicators: whether to show orbital angular momentum indicators at plaquette centers (OAM from single-plaquette loop current sum)
        oam_cmap: colormap for orbital angular momentum values
        oam_vmax: max absolute value for OAM colormap; if None, derived from data
        oam_marker_size: size of OAM indicator circles
        show_oam_direction_arrows: whether to show circular arrows indicating OAM direction
        oam_arrow_radius: radius of OAM circular arrows
        oam_arrow_lw: line width of OAM circular arrows
        oam_arrow_positive_color: color for positive OAM circular arrows
        oam_arrow_negative_color: color for negative OAM circular arrows
        oam_arrow_threshold: threshold for showing OAM circular arrows; if None, set to 1% of oam_vmax
        frame_texts: optional list of strings to use as title text per frame; if None, uses frame index "frame i/F"
    """
    fig, ax, ctx = _create_scene(
        lattice_frame_obs=lattice_frame_obs,
        geometry=geometry,
        density_cmap=density_cmap,
        density_vmin=density_vmin,
        density_vmax=density_vmax,
        current_max=current_max,
        site_marker_size=site_marker_size,
        show_flow_arrows=show_flow_arrows,
        arrows_per_edge=arrows_per_edge,
        arrow_scale=arrow_scale,
        arrow_width=arrow_width,
        arrow_color=arrow_color,
        show_oam_indicators=show_oam_indicators,
        oam_cmap=oam_cmap,
        oam_vmax=oam_vmax,
        oam_marker_size=oam_marker_size,
        show_oam_direction_arrows=show_oam_direction_arrows,
        oam_arrow_radius=oam_arrow_radius,
        oam_arrow_lw=oam_arrow_lw,
        oam_arrow_positive_color=oam_arrow_positive_color,
        oam_arrow_negative_color=oam_arrow_negative_color,
        oam_arrow_threshold=oam_arrow_threshold,
        frame_texts=frame_texts,
    )

    anim = animation.FuncAnimation(
        fig,
        lambda i: _update_scene(ctx, i),
        frames=ctx["F"],
        interval=1000 // max(1, fps),
        blit=False,
    )

    try:
        anim.save(out_path, writer="ffmpeg", fps=fps, dpi=dpi)
    except Exception:
        from matplotlib.animation import PillowWriter

        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)

    plt.close(fig)


def show_simulation_frame(
    lattice_frame_obs: LatticeFrameObservable,
    geometry: Lattice2DGeometry,
    frame: int = 0,
    density_cmap: str = "Greys",
    density_vmin: float = 0.0,
    density_vmax: float = 1.0,
    current_max: float | None = None,
    # Site circles size
    site_marker_size: float = 320.0,
    # Flow-direction arrows
    show_flow_arrows: bool = True,
    arrows_per_edge: int = 3,
    arrow_scale: float = 0.55,
    arrow_width: float = 0.04,
    arrow_color: str = "black",
    # OAM indicators
    show_oam_indicators: bool = True,
    oam_cmap: str = "RdBu",
    oam_vmax: float | None = None,
    oam_marker_size: float = 180.0,
    # curl direction circular arrows
    show_oam_direction_arrows: bool = True,
    oam_arrow_radius: float = 0.6,
    oam_arrow_lw: float = 1.5,
    oam_arrow_positive_color: str = "red",
    oam_arrow_negative_color: str = "blue",
    # Hide arrows for small curl values
    oam_arrow_threshold: float | None = None,
    # Optional per-frame text at top-left
    frame_texts: list[str] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a single frame to the current figure (useful for notebooks).

    Parameters:
        lattice_frame_obs: LatticeFrameObservable that recorded 'densities', 'currents', 'plaquette_oam' during the simulation
        geometry: Lattice2DGeometry providing site positions and nearest neighbor pairs
        frame: index of frame to render
        density_cmap: colormap for site densities
        density_vmin: min value for density colormap
        density_vmax: max value for density colormap
        current_max: max value for current colormap; if None, derived from data
        site_marker_size: size of site occupation circles
        show_flow_arrows: whether to show flow-direction arrows along bonds that indicate the current direction
        arrows_per_edge: number of arrows to draw along each bond
        arrow_scale: scaling factor for arrow lengths (w.r.t. current magnitude)
        arrow_width: width of arrows
        arrow_color: color of arrows
        show_oam_indicators: whether to show orbital angular momentum indicators at plaquette centers (OAM from single-plaquette loop current sum)
        oam_cmap: colormap for orbital angular momentum values
        oam_vmax: max absolute value for OAM colormap; if None, derived from data
        oam_marker_size: size of OAM indicator circles
        show_oam_direction_arrows: whether to show circular arrows indicating OAM direction
        oam_arrow_radius: radius of OAM circular arrows
        oam_arrow_lw: line width of OAM circular arrows
        oam_arrow_positive_color: color for positive OAM circular arrows
        oam_arrow_negative_color: color for negative OAM circular arrows
        oam_arrow_threshold: threshold for showing OAM circular arrows; if None, set to 1% of oam_vmax
        frame_texts: optional list of strings to use as title text per frame; if None, uses frame index "frame i/F"
        show: whether to call plt.show(). Useful if figure needs to be saved instead.
    
    Returns:
        fig, ax: the matplotlib Figure and Axes objects
    """
    fig, ax, ctx = _create_scene(
        lattice_frame_obs=lattice_frame_obs,
        geometry=geometry,
        density_cmap=density_cmap,
        density_vmin=density_vmin,
        density_vmax=density_vmax,
        current_max=current_max,
        site_marker_size=site_marker_size,
        show_flow_arrows=show_flow_arrows,
        arrows_per_edge=arrows_per_edge,
        arrow_scale=arrow_scale,
        arrow_width=arrow_width,
        arrow_color=arrow_color,
        show_oam_indicators=show_oam_indicators,
        oam_cmap=oam_cmap,
        oam_vmax=oam_vmax,
        oam_marker_size=oam_marker_size,
        show_oam_direction_arrows=show_oam_direction_arrows,
        oam_arrow_radius=oam_arrow_radius,
        oam_arrow_lw=oam_arrow_lw,
        oam_arrow_positive_color=oam_arrow_positive_color,
        oam_arrow_negative_color=oam_arrow_negative_color,
        oam_arrow_threshold=oam_arrow_threshold,
        frame_texts=frame_texts,
    )

    frame_clamped = int(np.clip(frame, 0, ctx["F"] - 1))
    _update_scene(ctx, frame_clamped)

    if show:
        plt.show()

    return fig, ax
