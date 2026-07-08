#!/usr/bin/env python3
"""Regenerate paper Figure 3 (Cuba case study: per-cycle and per-fluid
medians of net power and LCOE, plus Pareto frontier of all configurations).

Reads the ``Data_Results/Cuba/Time_series_data_Cuba_*.h5`` files produced by
``otex.regional.run_regional_analysis`` and renders a 5-panel figure that
matches the paper layout (a: Power by Cycle, b: LCOE by Cycle, c: Power by
Working Fluid, d: LCOE by Working Fluid, e: Pareto Frontier).

Notes
-----
* The local dataset is at ``136 MW gross`` (the pyOTEC/OTEX legacy default),
  not the ``100 MW`` that the current paper caption reports. This script
  regenerates from the actual data; the caption / text of the manuscript
  should be updated accordingly.
* Kalina and Uehara run on an ammonia–water zeotropic mixture — the
  filename token is ``ammonia`` but the figure labels them as
  ``NH3-H2O`` to match Figure 1's working-fluid panel and the paper text.
* Rankine Open flash-evaporates seawater — labelled ``Sea Water``.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = Path("/home/manuel/Dropbox/Net_zero_horizon/OTEX_main/"
                "Data_Results/Cuba")
OUT_DIR = Path("/home/manuel/Dropbox/Papers/Firs_author/14_OTEX/"
               "02_OTEX_paper")

CYCLE_ORDER = ['Kalina', 'Uehara',
               'Rankine Closed', 'Rankine Hybrid', 'Rankine Open']
FLUID_ORDER = ['NH3-H2O', 'NH3', 'Sea Water',
               'R134a', 'R245fa', 'Propane', 'Isobutane']

CYCLE_COLOR = {
    'Kalina':         '#e41a1c',   # red
    'Uehara':         '#4daf4a',   # green
    'Rankine Closed': '#377eb8',   # blue
    'Rankine Hybrid': '#ff7f00',   # orange
    'Rankine Open':   '#984ea3',   # purple
}
FLUID_COLOR = {
    'NH3-H2O':   '#e41a1c',
    'NH3':       '#377eb8',
    'Sea Water': '#984ea3',
    'R134a':     '#ff7f00',
    'R245fa':    '#a65628',
    'Propane':   '#4daf4a',
    'Isobutane': '#00897b',
}

FLUID_MAP = {
    'ammonia':   'NH3',
    'r134a':     'R134a',
    'r245fa':    'R245fa',
    'propane':   'Propane',
    'isobutane': 'Isobutane',
    'seawater':  'Sea Water',
}


def parse_config(fn: str) -> tuple[str, str, str] | None:
    """cycle, fluid, install from a Time_series_data_Cuba_<...>.h5 basename."""
    tag = (fn.replace('Time_series_data_Cuba_', '').replace('.h5', ''))
    tok = tag.split('_')
    if tok[0] == 'kalina':
        return 'Kalina', 'NH3-H2O', tok[2]
    if tok[0] == 'uehara':
        return 'Uehara', 'NH3-H2O', tok[2]
    if tok[0] == 'rankine' and tok[1] == 'closed':
        return 'Rankine Closed', FLUID_MAP[tok[2]], tok[3]
    if tok[0] == 'rankine' and tok[1] == 'hybrid':
        return 'Rankine Hybrid', FLUID_MAP[tok[2]], tok[3]
    if tok[0] == 'rankine' and tok[1] == 'open':
        return 'Rankine Open', FLUID_MAP[tok[2]], tok[3]
    return None


def load_records() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(DATA_DIR / 'Time_series_data_Cuba_*.h5'))):
        conf = parse_config(os.path.basename(f))
        if conf is None:
            continue
        cycle, fluid, install = conf
        with h5py.File(f, 'r') as h:
            lcoe = h['LCOE_nom/block0_values'][:].flatten()   # ct/kWh
            p_nom = h['p_net_nom/block0_values'][:].flatten() # kW (signed)
        pnet_mw = np.abs(p_nom) / 1000.0
        mask = np.isfinite(lcoe) & np.isfinite(pnet_mw) & (lcoe > 0) & (pnet_mw > 0)
        for l, p in zip(lcoe[mask], pnet_mw[mask]):
            rows.append({'cycle': cycle, 'fluid': fluid,
                         'install': install, 'lcoe': float(l),
                         'pnet': float(p)})
    return pd.DataFrame(rows)


def pareto_mask(pts: np.ndarray) -> np.ndarray:
    """pts (N, 2) with columns (pnet, lcoe). Pareto: maximise pnet,
    minimise lcoe. Return boolean mask of non-dominated points."""
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1]
                    and (pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1])):
                keep[i] = False
                break
    return keep


def scatter_by(ax, agg, cat_col, cat_order, y_col, y_label, colors, title):
    for xi, cat in enumerate(cat_order):
        sub = agg[agg[cat_col] == cat]
        for _, r in sub.iterrows():
            marker = '^' if r['install'] == 'offshore' else 'o'
            ax.scatter(xi, r[y_col], marker=marker, s=130,
                       color=colors[cat], edgecolor='black',
                       linewidth=0.7, alpha=0.9, zorder=3)
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_order, rotation=20, ha='right')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)


def main() -> None:
    plt.rcParams.update({
        'font.size': 11,
        'axes.linewidth': 1.1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    df = load_records()
    if df.empty:
        raise SystemExit(f"no data found under {DATA_DIR}")
    print(f"loaded {len(df)} site-records "
          f"across {df.groupby(['cycle','fluid','install']).ngroups} "
          f"(cycle,fluid,install) combinations")

    # Per-(cycle, fluid, install) medians — used by panel (e) so each
    # configuration is its own dot in the Pareto scatter.
    agg = (df.groupby(['cycle', 'fluid', 'install'])
             .agg(med_lcoe=('lcoe', 'median'),
                  med_pnet=('pnet', 'median'),
                  n=('lcoe', 'size'))
             .reset_index())

    # Per-(cycle, install) medians — one point per (cycle, install) so
    # panels (a, b) show a single triangle + circle per cycle, not one
    # marker per (cycle, fluid, install) combination.
    agg_cycle = (df.groupby(['cycle', 'install'])
                   .agg(med_lcoe=('lcoe', 'median'),
                        med_pnet=('pnet', 'median'))
                   .reset_index())

    # Per-(fluid, install) medians — same idea for panels (c, d).
    agg_fluid = (df.groupby(['fluid', 'install'])
                   .agg(med_lcoe=('lcoe', 'median'),
                        med_pnet=('pnet', 'median'))
                   .reset_index())

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[1, 1, 1.25],
                          hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    scatter_by(ax_a, agg_cycle, 'cycle', CYCLE_ORDER, 'med_pnet',
               'Median Net Power (MW)', CYCLE_COLOR, 'a) Power by Cycle')
    scatter_by(ax_b, agg_cycle, 'cycle', CYCLE_ORDER, 'med_lcoe',
               'Median LCOE (¢/kWh)', CYCLE_COLOR, 'b) LCOE by Cycle')
    scatter_by(ax_c, agg_fluid, 'fluid', FLUID_ORDER, 'med_pnet',
               'Median Net Power (MW)', FLUID_COLOR,
               'c) Power by Working Fluid')
    scatter_by(ax_d, agg_fluid, 'fluid', FLUID_ORDER, 'med_lcoe',
               'Median LCOE (¢/kWh)', FLUID_COLOR,
               'd) LCOE by Working Fluid')

    # --- Pareto frontier
    for _, r in agg.iterrows():
        marker = '^' if r['install'] == 'offshore' else 'o'
        ax_e.scatter(r['med_pnet'], r['med_lcoe'],
                     marker=marker, s=170,
                     color=CYCLE_COLOR[r['cycle']],
                     edgecolor='black', linewidth=0.7,
                     alpha=0.9, zorder=3)

    pts = agg[['med_pnet', 'med_lcoe']].values
    keep = pareto_mask(pts)
    pf = pts[keep]
    pf = pf[np.argsort(pf[:, 0])]
    ax_e.plot(pf[:, 0], pf[:, 1], 'k--', linewidth=1.8,
              alpha=0.85, zorder=2)

    # Label each Pareto-frontier point with its config — use adjustText so
    # they don't overlap the markers or each other.
    from adjustText import adjust_text
    texts = []
    for i, r in enumerate(agg.itertuples(index=False)):
        if keep[i]:
            label = f"{r.cycle} · {r.fluid}\n({r.install})"
            texts.append(ax_e.text(
                r.med_pnet, r.med_lcoe, label,
                fontsize=9, fontweight='bold', alpha=0.95,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', alpha=0.85,
                          edgecolor='lightgray', linewidth=0.5)))
    if texts:
        adjust_text(texts, ax=ax_e,
                    expand=(1.5, 1.6),
                    force_text=(0.8, 1.0),
                    force_points=(0.4, 0.4),
                    arrowprops=dict(arrowstyle='-', color='gray',
                                    lw=0.5, alpha=0.6))

    ax_e.set_xlabel('Median Net Power (MW)', fontweight='bold')
    ax_e.set_ylabel('Median LCOE (¢/kWh)', fontweight='bold')
    ax_e.set_title('e) Pareto Frontier', fontsize=12,
                   fontweight='bold', loc='left')
    ax_e.grid(linestyle='--', alpha=0.4)

    # --- Legend
    handles = []
    for c in CYCLE_ORDER:
        handles.append(Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=CYCLE_COLOR[c],
                              markeredgecolor='black', markersize=11,
                              label=c))
    handles += [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=11, label='Offshore'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=11, label='Onshore'),
        Line2D([0], [0], color='k', linestyle='--',
               linewidth=1.8, label='Pareto frontier'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.005),
               ncol=5, fontsize=10, frameon=True,
               framealpha=0.95)

    fig.subplots_adjust(top=0.96, bottom=0.07,
                        left=0.08, right=0.97)

    for ext in ('png', 'svg', 'tif'):
        out = OUT_DIR / f"Figure_3.{ext}"
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"  wrote {out}")

    # --- Also dump the aggregated medians as CSV for the manuscript
    csv_path = OUT_DIR / 'Figure_3_medians.csv'
    agg.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")


if __name__ == '__main__':
    main()
