#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for OTEC Uncertainty Analysis.

Run Monte Carlo, Sobol, or Tornado analysis from the command line.

Examples:
    # Monte Carlo analysis
    python scripts/uncertainty_analysis.py --T_WW 28 --T_CW 5 --method monte-carlo --samples 1000

    # Sobol sensitivity analysis
    python scripts/uncertainty_analysis.py --T_WW 28 --T_CW 5 --method sobol --samples 512

    # Tornado diagram analysis
    python scripts/uncertainty_analysis.py --T_WW 28 --T_CW 5 --method tornado

    # Full analysis (all methods)
    python scripts/uncertainty_analysis.py --T_WW 28 --T_CW 5 --method all --samples 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='OTEC Uncertainty Analysis CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required parameters
    parser.add_argument('--T_WW', type=float, required=True,
                        help='Warm water temperature (°C)')
    parser.add_argument('--T_CW', type=float, required=True,
                        help='Cold water temperature (°C)')

    # Analysis method
    parser.add_argument('--method', type=str, default='monte-carlo',
                        choices=['monte-carlo', 'sobol', 'tornado', 'all'],
                        help='Analysis method (default: monte-carlo)')

    # Optional parameters
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--p_gross', type=float, default=-136000,
                        help='Gross power output in kW (default: -136000)')
    parser.add_argument('--cost_level', type=str, default='low_cost',
                        choices=['low_cost', 'high_cost'],
                        help='Cost scenario (default: low_cost)')
    parser.add_argument('--output', type=str, default='lcoe',
                        choices=['lcoe', 'net_power', 'capex'],
                        help='Output variable to analyze (default: lcoe)')

    # Output options
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--export', action='store_true',
                        help='Export full results bundle (CSV + JSON) to --output-dir')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output files (default: current)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress bars')

    args = parser.parse_args()

    # Validate temperature difference
    delta_T = args.T_WW - args.T_CW
    if delta_T < 15:
        print(f"Warning: Temperature difference ({delta_T:.1f}°C) is low for OTEC.")
        print("Typical values are 20-25°C.")

    # Import analysis modules
    try:
        from otex.analysis import (
            MonteCarloAnalysis, UncertaintyConfig,
            SobolAnalysis, TornadoAnalysis,
            plot_histogram, plot_tornado, plot_sobol_indices,
            create_summary_figure,
            export_analysis,
        )
    except ImportError as e:
        print(f"Error importing otex.analysis: {e}")
        print("Make sure OTEX is installed: pip install -e .")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    show_progress = not args.quiet
    mc_results = None
    tornado_results = None
    sobol_results = None

    # Monte Carlo Analysis
    if args.method in ('monte-carlo', 'all'):
        print(f"\n{'='*60}")
        print("MONTE CARLO ANALYSIS")
        print(f"{'='*60}")
        print(f"T_WW: {args.T_WW}°C, T_CW: {args.T_CW}°C")
        print(f"Samples: {args.samples}, Seed: {args.seed}")

        config = UncertaintyConfig(
            n_samples=args.samples,
            seed=args.seed,
            parallel=not args.no_parallel
        )

        mc = MonteCarloAnalysis(
            T_WW=args.T_WW,
            T_CW=args.T_CW,
            config=config,
            p_gross=args.p_gross,
            cost_level=args.cost_level
        )

        mc_results = mc.run(show_progress=show_progress)
        stats = mc_results.compute_statistics()

        print(f"\n{args.output.upper()} Statistics:")
        s = stats[args.output]
        print(f"  Mean: {s[f'{args.output}_mean']:.2f}")
        print(f"  Std:  {s[f'{args.output}_std']:.2f}")
        print(f"  CV:   {s[f'{args.output}_cv']:.2%}")
        print(f"  Min:  {s[f'{args.output}_min']:.2f}")
        print(f"  Max:  {s[f'{args.output}_max']:.2f}")
        print(f"  P5:   {s[f'{args.output}_p5']:.2f}")
        print(f"  P50:  {s[f'{args.output}_median']:.2f}")
        print(f"  P95:  {s[f'{args.output}_p95']:.2f}")
        print(f"  90% CI: [{s[f'{args.output}_p5']:.2f}, {s[f'{args.output}_p95']:.2f}]")

        if args.save_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_histogram(mc_results, output=args.output, ax=ax)
            fig.savefig(output_dir / f'mc_histogram_{args.output}.png', dpi=150)
            plt.close(fig)
            print(f"  Saved: mc_histogram_{args.output}.png")

    # Tornado Analysis
    if args.method in ('tornado', 'all'):
        print(f"\n{'='*60}")
        print("TORNADO ANALYSIS")
        print(f"{'='*60}")

        tornado = TornadoAnalysis(
            T_WW=args.T_WW,
            T_CW=args.T_CW,
            p_gross=args.p_gross,
            cost_level=args.cost_level
        )

        tornado_results = tornado.run(output=args.output, show_progress=show_progress)

        print(f"\nParameter Sensitivity Ranking ({args.output.upper()}):")
        print(f"Baseline: {tornado_results.baseline:.2f}")
        print("-" * 50)
        for i, (name, swing) in enumerate(tornado_results.get_ranking()[:10], 1):
            print(f"  {i:2d}. {name:35s} swing: {swing:+.2f}")

        if args.save_plots:
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tornado(tornado_results, ax=ax)
            fig.savefig(output_dir / f'tornado_{args.output}.png', dpi=150)
            plt.close(fig)
            print(f"  Saved: tornado_{args.output}.png")

    # Sobol Analysis
    if args.method in ('sobol', 'all'):
        print(f"\n{'='*60}")
        print("SOBOL SENSITIVITY ANALYSIS")
        print(f"{'='*60}")

        try:
            sobol = SobolAnalysis(
                T_WW=args.T_WW,
                T_CW=args.T_CW,
                n_samples=min(args.samples, 512),
                p_gross=args.p_gross,
                cost_level=args.cost_level
            )

            sobol_results = sobol.run(output=args.output, show_progress=show_progress)

            print(f"\nSobol Indices ({args.output.upper()}):")
            print("-" * 60)
            print(f"{'Parameter':35s} {'S1':>8s} {'ST':>8s}")
            print("-" * 60)
            for name, st in sobol_results.get_ranking():
                idx = sobol_results.parameter_names.index(name)
                s1 = sobol_results.S1[idx]
                print(f"{name:35s} {s1:8.3f} {st:8.3f}")

            if args.save_plots:
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_sobol_indices(sobol_results, ax=ax)
                fig.savefig(output_dir / f'sobol_{args.output}.png', dpi=150)
                plt.close(fig)
                print(f"  Saved: sobol_{args.output}.png")

        except ImportError:
            print("SALib not installed. Skipping Sobol analysis.")
            print("Install with: pip install SALib")

    # Summary figure for 'all' method
    if args.method == 'all' and args.save_plots:
        if mc_results is not None and tornado_results is not None:
            fig = create_summary_figure(
                mc_results, tornado_results, sobol_results,
                output=args.output
            )
            fig.savefig(output_dir / f'summary_{args.output}.png', dpi=150)
            plt.close(fig)
            print(f"\nSaved: summary_{args.output}.png")

    # Full results export
    if args.export:
        print(f"\n{'='*60}")
        print("EXPORTING RESULTS")
        print(f"{'='*60}")
        export_analysis(
            output_dir=output_dir,
            mc_results=mc_results,
            tornado_results=tornado_results,
            sobol_results=sobol_results,
            metadata={
                'T_WW':       args.T_WW,
                'T_CW':       args.T_CW,
                'delta_T':    args.T_WW - args.T_CW,
                'p_gross_kW': args.p_gross,
                'cost_level': args.cost_level,
                'method':     args.method,
                'output_var': args.output,
                'seed':       args.seed,
            },
        )

    print(f"\n{'='*60}")
    print("Analysis complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
