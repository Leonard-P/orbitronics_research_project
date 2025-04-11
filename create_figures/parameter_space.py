import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.cm as cm # Import colormaps module

# 1. Setup Styling (Modernized)
plt.style.use('seaborn-v0_8-ticks') # A clean style with ticks
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
# Use Computer Modern for a classic LaTeX look, often preferred over Times
plt.rcParams['font.serif'] = 'Computer Modern Roman'
plt.rcParams['axes.labelsize'] = 14 # Slightly larger labels
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6.5) # Adjust figure size for better aspect

grid_resolution = 500

# 2. Define Axis Limits and Grid
x_max = 4 # Max value for L_y / l_WS
eta_max = 4 # Max value for omega / omega_R
# Increase grid density slightly for smoother gradients
x_grid = np.linspace(0, x_max, grid_resolution)
eta_grid = np.linspace(0, eta_max, grid_resolution)
X, ETA = np.meshgrid(x_grid, eta_grid)

# 3. Create Figure and Axes
fig, ax = plt.subplots()

# 4. Define Colors & Gradient Strategy
# Using perceptually uniform colormaps for professional look + visual appeal
# Viridis, Plasma, Magma are good choices
cmap_low = cm.viridis_r
cmap_mid = cm.autumn_r
cmap_high = cm.PuRd

# We'll create a gradient based on the eta value (vertical gradient)
# You could also use X for horizontal, or sqrt(X**2 + ETA**2) for radial etc.
Z_gradient = 0.7*np.sqrt(X**2+ETA**2)

# Define alpha for the gradient fills
alpha_fill = 0.5

# 5. Plot the Regions using pcolormesh with masks for gradients

# --- Define Masks for each region ---
mask_low = (X <= 1) & (ETA >= 1)
mask_mid = (X >= 1) & (ETA >= X) # eta >= x automatically implies eta >= 1 for x >= 1
mask_high = (X >= 1) & (ETA >= 1) & (ETA < X)

# --- Plot masked gradients ---
# Set vmin/vmax for consistent color mapping across regions based on eta
vmin_grad = -1 # Start gradient color mapping from eta=1
vmax_grad = eta_max

im_low = ax.pcolormesh(X, ETA, np.ma.masked_where(~mask_low, Z_gradient),
                       cmap=cmap_low, alpha=alpha_fill,
                       vmin=vmin_grad, vmax=vmax_grad, lw=0)

im_mid = ax.pcolormesh(X, ETA, np.ma.masked_where(~mask_mid, Z_gradient),
                       cmap=cmap_mid, alpha=alpha_fill,
                       vmin=vmin_grad, vmax=vmax_grad, lw=0)

im_high = ax.pcolormesh(X, ETA, np.ma.masked_where(~mask_high, Z_gradient),
                        cmap=cmap_high, alpha=alpha_fill,
                        vmin=vmin_grad, vmax=vmax_grad, lw=0)

# --- Fill region below eta=1 ---
# Use a subtle, light gray fill for the omega < omega_R region
ax.fill_between([0, x_max], 0, 1, color='gray', alpha=0.15, zorder=0, lw=0)


# --- Boundary Lines (Plot on top with higher zorder) ---
line_color = 'black'
line_width = 1.5
zorder_lines = 5 # Ensure lines are above gradient fills

# Boundary x=1 (for eta >= 1)
ax.plot([1, 1], [1, eta_max], color=line_color, linestyle='-', lw=line_width, zorder=zorder_lines)
# Boundary eta=x (for x >= 1)
ax.plot(x_grid[x_grid >= 1], x_grid[x_grid >= 1], color=line_color, linestyle='-', lw=line_width, zorder=zorder_lines)
# Boundary eta=1 (for x >= 0) - Plot segment for x<1 and x>1 separately if needed, or full line
ax.plot([0, x_max], [1, 1], color=line_color, linestyle='-', lw=line_width, zorder=zorder_lines)


# 6. Add Text Annotations with Enhanced Readability
# PathEffects provide an outline for better contrast against gradients
outline_effect = [pe.withStroke(linewidth=2.5, foreground='white')]
text_opts = {'ha': 'left', 'va': 'center', 'fontsize': 13, 'color': 'black', 'zorder': 10} # Ensure text is on top

# Adjust positions slightly for centering within regions
ax.text(0.1, 2.5, r'\textbf{no localization}' + '\n' + r'{$(E < W/L_y)$}', **text_opts)
ax.text(1.1, 3.0, r'\textbf{localization within lattice}' + '\n' + r'{$(W/L_y < E < \eta W/L_y)$}', **text_opts)
ax.text(2.1, 1.4, r'\textbf{localization within oscillations}' + '\n' + r'{$(E > \eta W/L_y)$}', **text_opts)

# Label for the region below eta=1
ax.text(x_max / 2, 0.5, r'$\omega < \omega_R$', ha='center', va='center',
        fontsize=13, color='black', zorder=10) # No outline needed here


# 7. Customize Axes and Labels
ax.set_xlabel(r'$L_y/l_{WS} = E L_y / W$')
ax.set_ylabel(r'$\eta = \omega / \omega_R$')
ax.set_xlim(0, x_max)
ax.set_ylim(0, eta_max)

# Remove top and right spines for cleaner look (already done by style, but explicit)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Make bottom and left spines slightly thicker/darker if desired
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['bottom'].set_linewidth(1.5)

# Ensure ticks point outwards (often default with '-ticks' styles)
ax.tick_params(axis='both', direction='out')

# 8. Legend (Optional - Text labels are quite clear)
# If needed, create custom legend handles for the regions
# from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor=cmap_low(0.7), edgecolor=None, alpha=alpha_fill, label='Low E'), # Sample color
#                    Patch(facecolor=cmap_mid(0.7), edgecolor=None, alpha=alpha_fill, label='Mid E'),
#                    Patch(facecolor=cmap_high(0.7), edgecolor=None, alpha=alpha_fill, label='High E')]
# ax.legend(handles=legend_elements, loc='lower right', frameon=False, title=r'\textbf{Regimes}')


# 9. Final Touches & Save
plt.tight_layout(pad=0.5) # Add a little padding
plt.savefig('efield_regimes_plot_stunning.pdf', bbox_inches='tight', dpi=300)
plt.savefig('efield_regimes_plot_stunning.png', bbox_inches='tight', dpi=300) # Also save PNG
plt.show()