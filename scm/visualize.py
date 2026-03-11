"""
visualize.py  –  Matplotlib-based visualization for CCG zone analysis

All functions import matplotlib lazily (optional dependency).
Each function returns (fig, ax) and optionally saves to file.

Usage
-----
    from scm.visualize import plot_zone_map, plot_payoff_trajectory

    fig, ax = plot_zone_map(zone_grid, param1_grid, param2_grid,
                            output_file='zone_map.png')
"""

import numpy as np


def _get_plt():
    """Lazy import of matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save_and_return(fig, output_file):
    """Save figure if path given, return fig."""
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
    return fig


def _detect_zone_transitions(zone_labels):
    """Find indices where zone label changes (for marking on 1D plots)."""
    transitions = []
    for i in range(1, len(zone_labels)):
        if zone_labels[i] != zone_labels[i - 1]:
            transitions.append(i)
    return transitions


# ======================================================================
# 2D Zone Map
# ======================================================================

def plot_zone_map(zone_grid, param1_grid, param2_grid,
                  param1_name='param1', param2_name='param2',
                  title='CCG Zone Map', output_file=None,
                  figsize=(10, 8), cmap='tab20'):
    """
    Plot zone boundaries as a 2D heatmap colored by zone label.

    Parameters
    ----------
    zone_grid : (n1, n2) array of str
        Zone labels from ccg_zone_map.
    param1_grid, param2_grid : 1D arrays
        Parameter values (param1 = rows/y-axis, param2 = cols/x-axis).
    param1_name, param2_name : str
        Axis labels.
    title : str
        Plot title.
    output_file : str or None
        Save to this path if given.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    plt = _get_plt()

    # Map unique zone strings to integer codes
    unique_zones = sorted(set(zone_grid.ravel()))
    zone_to_code = {z: i for i, z in enumerate(unique_zones)}
    code_grid = np.array([[zone_to_code[zone_grid[i, j]]
                           for j in range(zone_grid.shape[1])]
                          for i in range(zone_grid.shape[0])])

    fig, ax = plt.subplots(figsize=figsize)

    # Use pcolormesh for the heatmap
    im = ax.pcolormesh(param2_grid, param1_grid, code_grid,
                       cmap=cmap, shading='nearest')

    # Draw zone boundaries via contour
    if len(unique_zones) > 1:
        # Use grid centers for contour
        p2_centers = param2_grid
        p1_centers = param1_grid
        ax.contour(p2_centers, p1_centers, code_grid,
                   levels=np.arange(len(unique_zones) + 1) - 0.5,
                   colors='black', linewidths=1.0)

    ax.set_xlabel(param2_name, fontsize=12)
    ax.set_ylabel(param1_name, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Legend: colorbar with zone labels
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(unique_zones)))
    # Truncate long labels for display
    short_labels = [z[:30] for z in unique_zones]
    cbar.ax.set_yticklabels(short_labels, fontsize=8)
    cbar.set_label('Zone')

    _save_and_return(fig, output_file)
    return fig, ax


def plot_zone_map_with_payoff(zone_grid, payoff_grid, param1_grid, param2_grid,
                               player=0, param1_name='param1',
                               param2_name='param2', title=None,
                               output_file=None, figsize=(12, 5)):
    """
    Side-by-side: zone map + payoff heatmap for a given player.

    Parameters
    ----------
    zone_grid : (n1, n2) array of str
    payoff_grid : (n1, n2, m) array
    player : int
        Which player's payoff to show.
    """
    plt = _get_plt()

    if title is None:
        title = f'Zone Map + Class {player} Payoff'

    unique_zones = sorted(set(zone_grid.ravel()))
    zone_to_code = {z: i for i, z in enumerate(unique_zones)}
    code_grid = np.array([[zone_to_code[zone_grid[i, j]]
                           for j in range(zone_grid.shape[1])]
                          for i in range(zone_grid.shape[0])])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: zone map
    im1 = ax1.pcolormesh(param2_grid, param1_grid, code_grid,
                         cmap='tab20', shading='nearest')
    if len(unique_zones) > 1:
        ax1.contour(param2_grid, param1_grid, code_grid,
                    levels=np.arange(len(unique_zones) + 1) - 0.5,
                    colors='black', linewidths=1.0)
    ax1.set_xlabel(param2_name, fontsize=11)
    ax1.set_ylabel(param1_name, fontsize=11)
    ax1.set_title('Zone Structure', fontsize=12)

    # Right: payoff heatmap
    pay = payoff_grid[:, :, player]
    im2 = ax2.pcolormesh(param2_grid, param1_grid, pay,
                         cmap='viridis', shading='nearest')
    # Overlay zone boundaries
    if len(unique_zones) > 1:
        ax2.contour(param2_grid, param1_grid, code_grid,
                    levels=np.arange(len(unique_zones) + 1) - 0.5,
                    colors='white', linewidths=0.8, linestyles='--')
    ax2.set_xlabel(param2_name, fontsize=11)
    ax2.set_ylabel(param1_name, fontsize=11)
    ax2.set_title(f'Class {player} Payoff', fontsize=12)
    fig.colorbar(im2, ax=ax2, label='Payoff')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_and_return(fig, output_file)
    return fig, (ax1, ax2)


# ======================================================================
# 1D Trajectory Plots
# ======================================================================

def plot_payoff_trajectory(param_vals, payoff_values, class_labels=None,
                           param_name='parameter', title='CCG Payoff Trajectory',
                           zone_labels=None, output_file=None, figsize=(10, 6)):
    """
    Plot payoff curves for each class over a 1D parameter sweep.

    Parameters
    ----------
    param_vals : (n_points,) array
    payoff_values : (n_points, m) array
    class_labels : list of str or None
    param_name : str
    zone_labels : list of str or None
        Zone label at each point. If given, vertical lines mark zone transitions.
    output_file : str or None
    """
    plt = _get_plt()
    m = payoff_values.shape[1]
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(m)]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(m):
        ax.plot(param_vals, payoff_values[:, i], 'o-',
                label=class_labels[i], linewidth=2, markersize=3)

    # Mark zone transitions
    if zone_labels is not None:
        transitions = _detect_zone_transitions(zone_labels)
        for t in transitions:
            x_mid = (param_vals[t - 1] + param_vals[t]) / 2
            ax.axvline(x_mid, color='gray', linestyle='--', linewidth=0.8,
                       alpha=0.7)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Payoff', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _save_and_return(fig, output_file)
    return fig, ax


def plot_wage_trajectory(param_vals, wage_values, class_labels=None,
                         param_name='parameter', title='CCG Wage Trajectory',
                         zone_labels=None, output_file=None, figsize=(10, 6)):
    """Plot wage curves for each class over a 1D parameter sweep."""
    plt = _get_plt()
    m = wage_values.shape[1]
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(m)]

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(m):
        ax.plot(param_vals, wage_values[:, i], 's-',
                label=class_labels[i], linewidth=2, markersize=3)

    if zone_labels is not None:
        transitions = _detect_zone_transitions(zone_labels)
        for t in transitions:
            x_mid = (param_vals[t - 1] + param_vals[t]) / 2
            ax.axvline(x_mid, color='gray', linestyle='--', linewidth=0.8,
                       alpha=0.7)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Wage Income', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _save_and_return(fig, output_file)
    return fig, ax


def plot_price_trajectory(param_vals, price_values, good_labels=None,
                          param_name='parameter', title='CCG Price Trajectory',
                          zone_labels=None, output_file=None, figsize=(10, 6)):
    """Plot price curves for each good over a 1D parameter sweep."""
    plt = _get_plt()
    n = price_values.shape[1]
    if good_labels is None:
        good_labels = [f'Good {j}' for j in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    for j in range(n):
        ax.plot(param_vals, price_values[:, j], '^-',
                label=good_labels[j], linewidth=2, markersize=3)

    if zone_labels is not None:
        transitions = _detect_zone_transitions(zone_labels)
        for t in transitions:
            x_mid = (param_vals[t - 1] + param_vals[t]) / 2
            ax.axvline(x_mid, color='gray', linestyle='--', linewidth=0.8,
                       alpha=0.7)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    _save_and_return(fig, output_file)
    return fig, ax


# ======================================================================
# Allocation Pattern
# ======================================================================

def plot_allocation_pattern(X, class_labels=None, good_labels=None,
                            title='Goods Allocation by Class',
                            output_file=None, figsize=(10, 6)):
    """
    Stacked bar chart: goods consumed by each class.

    Parameters
    ----------
    X : (m, n) array
        Allocation matrix (units).
    class_labels : list or None
    good_labels : list or None
    """
    plt = _get_plt()
    m, n = X.shape
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(m)]
    if good_labels is None:
        good_labels = [f'Good {j}' for j in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(m)
    width = 0.6
    bottom = np.zeros(m)

    colors = plt.cm.Set2(np.linspace(0, 1, n))
    for j in range(n):
        ax.bar(x, X[:, j], width, label=good_labels[j],
               bottom=bottom, color=colors[j])
        bottom += X[:, j]

    ax.set_ylabel('Allocation (units)', fontsize=12)
    ax.set_xlabel('Labour Class', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))

    fig.tight_layout()
    _save_and_return(fig, output_file)
    return fig, ax


# ======================================================================
# Forest Diagram (Bipartite Graph)
# ======================================================================

def plot_forest_diagram(X, I, J, class_labels=None, good_labels=None,
                        title='Fisher Forest (Spending Pattern)',
                        output_file=None, figsize=(8, 6)):
    """
    Bipartite graph: classes on left, goods on right, edges = spending.
    Edge width proportional to allocation.

    Parameters
    ----------
    X : (m, n) array
        Allocation matrix (units).
    I : array of int
        Active class indices.
    J : array of int
        Active good indices.
    """
    plt = _get_plt()
    from matplotlib.patches import FancyArrowPatch

    I = np.asarray(I).ravel()
    J = np.asarray(J).ravel()
    m_active = len(I)
    n_active = len(J)

    if class_labels is None:
        class_labels = [f'L{i}' for i in range(X.shape[0])]
    if good_labels is None:
        good_labels = [f'g{j}' for j in range(X.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)

    # Position classes on left (x=0), goods on right (x=1)
    class_y = np.linspace(0, 1, max(m_active, 2))[:m_active]
    good_y = np.linspace(0, 1, max(n_active, 2))[:n_active]

    # Draw edges (allocation flows)
    max_alloc = max(X[I][:, J].max(), 1e-6)
    for ki, i in enumerate(I):
        for kj, j in enumerate(J):
            if X[i, j] > 1e-6:
                width = 0.5 + 4.0 * (X[i, j] / max_alloc)
                alpha = 0.3 + 0.6 * (X[i, j] / max_alloc)
                ax.plot([0, 1], [class_y[ki], good_y[kj]],
                        '-', color='steelblue', linewidth=width, alpha=alpha)
                # Label the flow
                mid_x = 0.5
                mid_y = (class_y[ki] + good_y[kj]) / 2
                ax.text(mid_x, mid_y, f'{X[i, j]:.2f}',
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8))

    # Draw class nodes
    for ki, i in enumerate(I):
        ax.plot(0, class_y[ki], 'o', markersize=20, color='coral',
                zorder=5)
        ax.text(-0.08, class_y[ki], class_labels[i],
                fontsize=11, ha='right', va='center', fontweight='bold')

    # Draw good nodes
    for kj, j in enumerate(J):
        ax.plot(1, good_y[kj], 's', markersize=20, color='lightgreen',
                zorder=5)
        ax.text(1.08, good_y[kj], good_labels[j],
                fontsize=11, ha='left', va='center', fontweight='bold')

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    fig.tight_layout()
    _save_and_return(fig, output_file)
    return fig, ax


# ======================================================================
# Gradient Field (Quiver Plot)
# ======================================================================

def plot_gradient_field(grad_grid, param1_grid, param2_grid,
                        param1_name='param1', param2_name='param2',
                        zone_grid=None, title='CCG Gradient Field',
                        output_file=None, figsize=(10, 8)):
    """
    Quiver plot of gradient direction overlaid on zone map.

    Parameters
    ----------
    grad_grid : (n1, n2, 2) array
        Gradient components at each grid point.
        grad_grid[i, j, 0] = ∂payoff/∂param1,
        grad_grid[i, j, 1] = ∂payoff/∂param2.
    param1_grid, param2_grid : 1D arrays
    zone_grid : (n1, n2) array of str or None
        If given, show zone boundaries behind the quiver plot.
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=figsize)

    # Background zone map if provided
    if zone_grid is not None:
        unique_zones = sorted(set(zone_grid.ravel()))
        zone_to_code = {z: i for i, z in enumerate(unique_zones)}
        code_grid = np.array([[zone_to_code[zone_grid[i, j]]
                               for j in range(zone_grid.shape[1])]
                              for i in range(zone_grid.shape[0])])
        ax.pcolormesh(param2_grid, param1_grid, code_grid,
                      cmap='Pastel1', shading='nearest', alpha=0.5)
        if len(unique_zones) > 1:
            ax.contour(param2_grid, param1_grid, code_grid,
                       levels=np.arange(len(unique_zones) + 1) - 0.5,
                       colors='gray', linewidths=0.8)

    # Quiver plot
    P2, P1 = np.meshgrid(param2_grid, param1_grid)
    U_comp = grad_grid[:, :, 1]  # horizontal (param2 direction)
    V_comp = grad_grid[:, :, 0]  # vertical (param1 direction)

    ax.quiver(P2, P1, U_comp, V_comp, color='darkred', alpha=0.8,
              scale_units='xy', angles='xy')

    ax.set_xlabel(param2_name, fontsize=12)
    ax.set_ylabel(param1_name, fontsize=12)
    ax.set_title(title, fontsize=14)

    _save_and_return(fig, output_file)
    return fig, ax
