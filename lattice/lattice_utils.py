from matplotlib import patheffects
import matplotlib.axis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from .lattice_geometry import LatticeGeometry


def plot_arrow(ax: matplotlib.axis, x: float, y: float, dx: float, dy: float, color: str = "black", label: str = ""):
    # ax.annotate(
    #     "",
    #     xy=(x + dx, y + dy),
    #     xytext=(x, y),
    #     arrowprops=dict(arrowstyle="->", color=color, lw=4),
    # )

    ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, 
                fc=color, ec=color, width=0.02, zorder=4, length_includes_head=True)

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
        circle = plt.Circle((x, y), 0.3, facecolor=circle_color, edgecolor="black", zorder=2)

        ax.add_patch(circle)
        plt.plot(x, y, alpha=0)

        ax.text(
            x,
            y,
            f"{val:.5f}",
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


def plot_site_connections(
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


if __name__ == "__main__":
    grid_values = np.random.rand(25)
    conn_matrix = np.zeros((25, 25))
    for i in range(24):
        conn_matrix[i, i + 1] = np.random.rand() - 0.5

    fig, ax = plt.subplots(figsize=(10, 10))
    # plot_site_grid(grid_values, 5, 5, ax)
    # plot_site_connections(conn_matrix, 5, 5, ax)

    plt.tight_layout()
    plt.show()
