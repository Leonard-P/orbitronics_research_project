from matplotlib import patheffects
import matplotlib.axis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Arc, RegularPolygon

from .lattice_geometry import LatticeGeometry


def drawCirc(ax, radius, centX, centY, angle_, theta2_, color_="black", lw=1, anticlockwise=True):
    arc = Arc([centX, centY], radius, radius, angle=angle_, theta1=0, theta2=theta2_, capstyle="round", linestyle="-", lw=lw, color=color_)
    ax.add_patch(arc)

    if anticlockwise:
        orientation = np.radians(angle_ + theta2_)
    else:
        theta2_ = 0
        orientation = np.radians(angle_) + np.pi
    endX = centX + (radius / 2) * np.cos(np.radians(theta2_ + angle_))
    endY = centY + (radius / 2) * np.sin(np.radians(theta2_ + angle_))

    ax.add_patch(RegularPolygon((endX, endY), 3, radius=radius / 7, orientation=orientation, color=color_, zorder=10))


def plot_arrow(ax: matplotlib.axis, x: float, y: float, dx: float, dy: float, color: str = "black", label: str = ""):
    # ax.annotate(
    #     "",
    #     xy=(x + dx, y + dy),
    #     xytext=(x, y),
    #     arrowprops=dict(arrowstyle="->", color=color, lw=4),
    # )

    ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc=color, ec=color, width=0.02, zorder=4, length_includes_head=True)

    if label:
        ax.text(x + 0.2, y + 0.2, label, fontsize=14, color=color)

    # plot transparent dot at arrow end
    plt.plot(x + dx, y - dy, alpha=0)


def plot_site_grid(
    site_values: np.ndarray,
    geometry: LatticeGeometry,
    ax: matplotlib.axis = None,
    vmin: float = None,
    vmax: float = None,
    cmap_name: str = "inferno",
    print_text_labels: bool = True,
) -> matplotlib.axis:
    """Plot grid sites with values as colored circles."""
    Lx, Ly = geometry.dimensions
    external_axis = ax is not None
    if not external_axis:
        _, ax = plt.subplots(figsize=(2 * Lx, 2 * Ly))

    norm = plt.Normalize(vmin=vmin if vmin is not None else np.nanmin(site_values), vmax=vmax if vmax is not None else np.nanmax(site_values))
    cmap = plt.get_cmap(cmap_name)

    # Plot nodes with their diagonal values.
    for idx, val in enumerate(site_values):
        if np.isnan(val):
            continue
        x, y = geometry.site_to_position(idx)
        color_val = complex(norm(val)).real
        circle_color = cmap(color_val)
        circle = plt.Circle((x, y), 0.2, facecolor=circle_color, edgecolor="black", zorder=2, lw=1.5)

        ax.add_patch(circle)
        plt.plot(x, y, alpha=0)

        if print_text_labels:
            ax.text(
                x,
                y,
                f"{val:.3f}",
                color="white",
                ha="center",
                va="center",
                fontsize=10,
                zorder=3,
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")],
            )

    ax.set_aspect("equal")
    ax.axis("off")
    if not external_axis:
        plt.show()

    return ax


def plot_site_connections_old(
    connection_matrix: np.ndarray,
    geometry: LatticeGeometry,
    ax: matplotlib.axis = None,
    max_flow: float = 1,
    label_connection_strength: bool = True,
    plot_flow_direction_arrows: bool = True,
) -> matplotlib.axis:
    """Plot connections between sites with arrows showing flow direction."""
    Lx, Ly = geometry.dimensions
    if ax is None:
        _, ax = plt.subplots(figsize=(2 * Lx, 2 * Ly))

    N = connection_matrix.shape[0]
    for u in range(N):
        for v in range(u + 1, N):
            if (J := connection_matrix[u, v]) == 0:
                continue

            x1, y1 = geometry.site_to_position(u)
            x2, y2 = geometry.site_to_position(v)

            linewidth = 2 * abs(J) / max_flow
            color = "red" if J < 0 else "blue"
            color = "black"
            ax.plot([x1, x2], [y1, y2], "-", color=color, linewidth=5 * linewidth, zorder=1, alpha=0.3 if plot_flow_direction_arrows else 1)

            if label_connection_strength:
                xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(
                    xm,
                    ym + 0.1,
                    f"{J:.1f}",
                    color="red",
                    ha="center",
                    va="center",
                    fontsize=8,
                    zorder=4,
                )

            if plot_flow_direction_arrows:
                xval = np.linspace(x1, x2, 8)
                yval = np.linspace(y1, y2, 8)

                if x1 == x2:
                    arrow = "v" if J < 0 else "^"
                else:
                    arrow = ">" if J > 0 else "<"

                ax.plot(
                    xval,
                    yval,
                    arrow,
                    linewidth=linewidth,
                    color=color,
                    markersize=(4 + 4 * linewidth),
                    zorder=0,
                )

    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_site_connections(
    connection_matrix: np.ndarray,
    geometry: LatticeGeometry,
    ax: matplotlib.axes.Axes = None,
    max_flow: float = 1.0,
    label_connection_strength: bool = True,
    plot_flow_direction_arrows: bool = True,
) -> matplotlib.axes.Axes:
    """Plot connections between sites with arrows showing flow direction."""
    Lx, Ly = geometry.dimensions
    if ax is None:
        fig_width = max(1, 2 * int(Lx))  # Ensure dimensions are reasonable
        fig_height = max(1, 2 * int(Ly))
        _, ax = plt.subplots(figsize=(fig_width, fig_height))

    if max_flow <= 0:
        abs_J_values = np.abs(connection_matrix[connection_matrix != 0])
        if len(abs_J_values) > 0:
            max_flow = np.max(abs_J_values)
        if max_flow <= 0:  # Still no positive max_flow (e.g., all J are 0)
            max_flow = 1.0  # Default to 1 to avoid division by zero

    N = connection_matrix.shape[0]

    for u in range(N):
        for v in range(N):  # Iterate full matrix to handle J_uv and J_vu if they can be different
            if u == v:
                continue

            # Ensure each pair (u,v) is processed once based on your connection_matrix structure.
            # If connection_matrix is symmetric (J_uv = J_vu) and you only store one, iterate v from u+1.
            # If J_uv can be different from J_vu (e.g. representing net flow), this loop is fine.
            # Assuming connection_matrix[u,v] means flow from u to v if J>0, v to u if J<0.
            # If J_uv represents J_vu (symmetric), then use range(u+1, N) for v.
            # The original code used range(u+1,N) implying J_uv is the only one to consider. Let's stick to that.
            if v <= u:  # Process each pair once, assuming J_uv is the value to use
                continue

            J = connection_matrix[u, v]
            if J == 0:
                continue

            x1, y1 = geometry.site_to_position(u)
            x2, y2 = geometry.site_to_position(v)

            # lw_j is a factor proportional to current, ranging 0 to 2 if max_flow is accurate
            lw_j = 2 * abs(J) / max_flow

            line_color = "black"  # Fixed color from your snippet

            # Plot the base connection line (semi-transparent if arrows are shown)
            ax.plot([x1, x2], [y1, y2], "-", color="black", linewidth=lw_j, zorder=1, alpha=1 if plot_flow_direction_arrows else 1)

            if label_connection_strength:
                xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                # Slightly offset label to avoid overlapping with line/arrows
                # Heuristic offset, may need adjustment based on typical line angles
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                offset_dx = -np.sin(angle_rad) * 0.15  # Perpendicular offset
                offset_dy = np.cos(angle_rad) * 0.15

                ax.text(
                    xm + offset_dx,
                    ym + offset_dy,
                    f"{J:.2f}",  # Format to two decimal places
                    color="black",  # Or a contrasting color
                    ha="center",
                    va="center",
                    fontsize=8,
                    zorder=4,  # Text on top of everything
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.1),  # Optional background for legibility
                )

            if plot_flow_direction_arrows:
                num_arrow_markers = 4  # Number of small arrows along the line segment
                # Adjust this for desired density. Original image suggests 3-5.

                # Determine arrow vector (dx_flow, dy_flow) which points in the direction of current
                if J > 0:  # Flow from u to v
                    dx_flow = x2 - x1
                    dy_flow = y2 - y1
                else:  # Flow from v to u
                    dx_flow = x1 - x2
                    dy_flow = y1 - y2

                length_flow = np.sqrt(dx_flow**2 + dy_flow**2)
                if length_flow == 0:  # Should not happen for u != v
                    continue

                # Normalized unit vector for flow direction
                udx_flow = dx_flow / length_flow
                udy_flow = dy_flow / length_flow

                lw_j = 2* max(0, lw_j)

                marker_length_data = length_flow / (num_arrow_markers + 1) / 2.2
                marker_length_data *= lw_j
                U_quiver = udx_flow * marker_length_data
                V_quiver = udy_flow * marker_length_data

                for k_marker in range(num_arrow_markers):
                    # Position of the center of the k-th arrow marker along the segment
                    frac = (k_marker + 1) / (num_arrow_markers + 1) - 0.09
                    px = x1 + frac * (x2 - x1)
                    py = y1 + frac * (y2 - y1)

                    length = marker_length_data
                    width = 0.1 * lw_j / num_arrow_markers
                    hal = hl = 1.0 / width * length

                    ax.quiver(
                        px,
                        py,
                        U_quiver,
                        V_quiver,
                        units="xy",
                        angles="xy",
                        scale_units="xy",
                        scale=1,
                        pivot="middle",  # (px,py) is the center of the arrow
                        color=line_color,  # Fill color of the arrow
                        headwidth=hl,
                        headlength=hl,
                        headaxislength=hal,
                        width=width,  # Shaft width in data units
                        linewidth=0,  # No edge line around the arrow shape
                        # minshaft=.01,  # Controls behavior for very short arrows (relative to head length)
                        zorder=0,
                    )

    ax.set_aspect("equal")
    ax.axis("off")
    return ax


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # add project root to path so package modules can be imported when running as script
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from lattice2d import Lattice2D, SimulationParameters
    from lattice_geometry import LatticeGeometry, HexagonalLatticeGeometry, RectangularLatticeGeometry
    from field_generator import FieldAmplitudeGenerator
    import numpy as np
    import matplotlib.pyplot as plt

    E = 1e-4
    omega = 4 / np.pi

    l = Lattice2D(
        HexagonalLatticeGeometry((13, 16)),
        SimulationParameters(
            t_hop=-1,
            E_amplitude=FieldAmplitudeGenerator.ramped_oscillation(E, omega, np.pi/omega),
            E_direction=np.array([0, 1]),
            h=0.01,
            T=5.85 * np.pi / omega,
            substeps=2,
        ),
    )

    l.evolve(first_snapshot_step=0 * 8 / 0.01, decay_time=10)
    # # plot_site_grid(grid_values, 5, 5, ax)
    # # plot_site_connections(conn_matrix, 5, 5, ax)
    _, ax = plt.subplots(figsize=(7, 8))
    l.plot_current_density(-1, ax, auto_normalize=False, curl_norm=0.000001)
    plt.savefig("test2.png", dpi=400, bbox_inches="tight")