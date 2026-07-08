#!/usr/bin/env python3
"""Regenerate paper Figure 2 (Cuba case study: ridgeline KDE distributions
of net power output and LCOE for every (cycle, working fluid, install)
combination).

Layout matches the docx original: 13 rows (one per cycle × fluid pair)
stacked from top to bottom in the fixed order

    Uehara - NH3-H2O
    Rankine Open - Sea Water
    Rankine Hybrid - R245fa
    Rankine Hybrid - R134a
    Rankine Hybrid - Propane
    Rankine Hybrid - Isobutane
    Rankine Hybrid - Ammonia
    Rankine Closed - R245fa
    Rankine Closed - R134a
    Rankine Closed - Propane
    Rankine Closed - Isobutane
    Rankine Closed - Ammonia
    Kalina - NH3-H2O

Two side-by-side panels (a: net power, b: LCOE). Each row shows two KDE
ridgelines — solid = offshore, dashed = onshore — colored by cycle.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

DATA_DIR = Path("/home/manuel/Dropbox/Net_zero_horizon/OTEX_main/"
                "Data_Results/Cuba")
OUT_DIR = Path("/home/manuel/Dropbox/Papers/Firs_author/14_OTEX/"
               "02_OTEX_paper")

# Row order — top-to-bottom in the docx original. We reverse it for
# matplotlib (bottom-up y-axis) so the first entry ends up at the top.
ROW_ORDER = [
    ('Uehara',         'NH3-H2O',   'uehara',         'ammonia'),
    ('Rankine Open',   'Sea Water', 'rankine_open',   'seawater'),
    ('Rankine Hybrid', 'R245fa',    'rankine_hybrid', 'r245fa'),
    ('Rankine Hybrid', 'R134a',     'rankine_hybrid', 'r134a'),
    ('Rankine Hybrid', 'Propane',   'rankine_hybrid', 'propane'),
    ('Rankine Hybrid', 'Isobutane', 'rankine_hybrid', 'isobutane'),
    ('Rankine Hybrid', 'Ammonia',   'rankine_hybrid', 'ammonia'),
    ('Rankine Closed', 'R245fa',    'rankine_closed', 'r245fa'),
    ('Rankine Closed', 'R134a',     'rankine_closed', 'r134a'),
    ('Rankine Closed', 'Propane',   'rankine_closed', 'propane'),
    ('Rankine Closed', 'Isobutane', 'rankine_closed', 'isobutane'),
    ('Rankine Closed', 'Ammonia',   'rankine_closed', 'ammonia'),
    ('Kalina',         'NH3-H2O',   'kalina',         'ammonia'),
]

CYCLE_COLOR = {
    'Kalina':         '#e41a1c',   # red
    'Uehara':         '#4daf4a',   # green
    'Rankine Closed': '#377eb8',   # blue
    'Rankine Hybrid': '#ff7f00',   # orange
    'Rankine Open':   '#984ea3',   # purple
}

# Layout parameters
ROW_STEP = 1.0                   # vertical spacing between ridgelines
RIDGE_SCALE = 0.75               # KDE peak amplitude relative to ROW_STEP


def h5_stats(path: str):
    """Return per-site (LCOE ¢/kWh, Pnet MW) as clean numpy arrays."""
    with h5py.File(path, 'r') as h:
        lcoe = h['LCOE_nom/block0_values'][:].flatten()
        pnet_kw = h['p_net_nom/block0_values'][:].flatten()
    pnet = np.abs(pnet_kw) / 1000.0
    ok = np.isfinite(lcoe) & np.isfinite(pnet) & (lcoe > 0) & (pnet > 0)
    return lcoe[ok], pnet[ok]


def kde_curve(vals: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Scaled KDE so its peak equals RIDGE_SCALE (or return zeros when
    the sample is too degenerate for a KDE)."""
    if len(vals) < 5 or np.std(vals) < 1e-6:
        return np.zeros_like(x_grid)
    kde = gaussian_kde(vals, bw_method='scott')
    y = kde(x_grid)
    peak = float(y.max())
    if peak <= 0:
        return np.zeros_like(x_grid)
    return y * (RIDGE_SCALE / peak)


def draw_row(ax, y_center: float, x_grid: np.ndarray,
             vals_off: np.ndarray, vals_on: np.ndarray, color: str) -> None:
    """Draw two ridgelines for one config (offshore solid + onshore
    dashed), both baselined at ``y_center``."""
    if vals_off.size:
        y = kde_curve(vals_off, x_grid) + y_center
        ax.fill_between(x_grid, y_center, y, color=color, alpha=0.55,
                        zorder=3)
        ax.plot(x_grid, y, color=color, linewidth=1.4, zorder=4)
    if vals_on.size:
        y = kde_curve(vals_on, x_grid) + y_center
        ax.fill_between(x_grid, y_center, y, color=color, alpha=0.18,
                        zorder=2)
        ax.plot(x_grid, y, color=color, linewidth=1.2,
                linestyle='--', dashes=(4, 2.5), zorder=4)
    ax.axhline(y_center, color='#cccccc', linewidth=0.4, zorder=1)


def flat_path(cyc: str, fl: str, inst: str) -> str:
    return str(DATA_DIR / f"Time_series_data_Cuba_{cyc}_{fl}_"
                          f"{inst}_2023_136.0_MW_low_cost.h5")


def main() -> None:
    plt.rcParams.update({
        'font.size': 11,
        'axes.linewidth': 1.0,
    })

    # Collect data
    per_row = []   # (display, cycle, fluid, values_lcoe_off/on, pnet_off/on)
    for cyc_disp, fl_disp, cyc_key, fl_key in ROW_ORDER:
        off = flat_path(cyc_key, fl_key, 'offshore')
        on  = flat_path(cyc_key, fl_key, 'onshore')
        l_off, p_off = h5_stats(off) if os.path.isfile(off) \
                       else (np.array([]), np.array([]))
        l_on, p_on = h5_stats(on) if os.path.isfile(on) \
                     else (np.array([]), np.array([]))
        per_row.append((cyc_disp, fl_disp, cyc_key,
                        l_off, l_on, p_off, p_on))
        print(f"  {cyc_disp:16s} {fl_disp:10s}  "
              f"offshore n={len(l_off):3d}  onshore n={len(l_on):3d}")

    # Common x grids from the pooled data (padded)
    all_lcoe = np.concatenate([np.concatenate([r[3], r[4]])
                               for r in per_row])
    all_pnet = np.concatenate([np.concatenate([r[5], r[6]])
                               for r in per_row])
    lcoe_lo, lcoe_hi = float(all_lcoe.min()) * 0.9, float(all_lcoe.max()) * 1.02
    pnet_lo, pnet_hi = float(all_pnet.min()) * 0.9, float(all_pnet.max()) * 1.02
    x_lcoe = np.linspace(lcoe_lo, lcoe_hi, 500)
    x_pnet = np.linspace(pnet_lo, pnet_hi, 500)

    # ---- Figure
    fig, (ax_p, ax_l) = plt.subplots(1, 2, figsize=(13, 8), sharey=True)

    # Rows are laid out with the FIRST entry at the top → iterate in
    # reverse so higher y = earlier row.
    row_labels = []
    for i, (cyc_disp, fl_disp, cyc_key,
            l_off, l_on, p_off, p_on) in enumerate(per_row):
        y_center = (len(per_row) - 1 - i) * ROW_STEP
        color = CYCLE_COLOR[cyc_disp]
        row_labels.append((y_center, f"{cyc_disp} - {fl_disp}", color))
        draw_row(ax_p, y_center, x_pnet, p_off, p_on, color)
        draw_row(ax_l, y_center, x_lcoe, l_off, l_on, color)

    # Y ticks and labels
    ticks = [rl[0] for rl in row_labels]
    labels = [rl[1] for rl in row_labels]
    for ax in (ax_p, ax_l):
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.5, len(per_row) - 1 + 1.0)
        ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Colour tick labels by cycle
    for y_c, lbl, col in row_labels:
        for tick_label in ax_p.get_yticklabels():
            if tick_label.get_text() == lbl:
                tick_label.set_color(col)
                tick_label.set_fontweight('bold')

    ax_p.set_xlabel('Net Power (MW)', fontweight='bold')
    ax_l.set_xlabel('LCOE (¢/kWh)', fontweight='bold')
    ax_p.set_title('a) Net Power Distribution', fontsize=13,
                   fontweight='bold', loc='left')
    ax_l.set_title('b) LCOE Distribution', fontsize=13,
                   fontweight='bold', loc='left')

    # ---- Legend
    cycle_handles = [Patch(facecolor=CYCLE_COLOR[c], edgecolor='black',
                           linewidth=0.5, label=c)
                     for c in ('Kalina', 'Uehara', 'Rankine Closed',
                               'Rankine Hybrid', 'Rankine Open')]
    style_handles = [
        Line2D([0], [0], color='#555555', linewidth=2.2,
               label='Offshore (KDE)'),
        Line2D([0], [0], color='#555555', linewidth=2.2,
               linestyle='--', dashes=(4, 2.5), label='Onshore (KDE)'),
    ]
    fig.legend(handles=cycle_handles + style_handles,
               loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=7, fontsize=10, frameon=True,
               framealpha=0.95, columnspacing=1.6)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.94,
                        bottom=0.14, wspace=0.05)

    for ext in ('png', 'svg', 'tif'):
        out = OUT_DIR / f"Figure_2.{ext}"
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  wrote {out}")


if __name__ == '__main__':
    main()
