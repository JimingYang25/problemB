"""
Generate figures for the paper:
  - figure_cross_sections.png: 9 cross-sections overlaid
  - figure_evolution.png: A(t), ξ(t), H(t) evolution curves
  - figure_regression.png: Regression diagnostic plots
  - figure_loocv.png: LOOCV prediction vs actual
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates

# Setup Chinese font
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'


def plot_cross_sections(sections: dict, output_path: str):
    """Plot all 9 cross-sections overlaid, with the reference water level."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color cycle
    cmap = plt.cm.viridis
    dates = sorted(sections.keys())
    n = len(dates)

    # Find global x range
    all_x = []
    for df in sections.values():
        all_x.extend(df['起点距离(m)'].values)
    x_min, x_max = min(all_x), max(all_x)

    for i, date in enumerate(dates):
        df = sections[date]
        x = df['起点距离(m)'].values
        z = df['河底高程(m)'].values
        color = cmap(i / max(1, n - 1))
        ax.plot(x, z, color=color, linewidth=1.0, alpha=0.8, label=f'{date}')

    # Reference water level
    ax.axhline(y=43.0, color='dodgerblue', linestyle='--', linewidth=2,
               alpha=0.7, label='Reference WL (Z=43m)')

    ax.set_xlabel('起点距离 (m)', fontsize=12)
    ax.set_ylabel('河底高程 (m)', fontsize=12)
    ax.set_title('9 Cross-Section Measurements (2016–2021)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved cross-section plot → {output_path}")


def plot_evolution(
    indicators_df: pd.DataFrame,
    ode_results: dict,
    output_path: str
):
    """Plot A(t), ξ(t), H(t) evolution with fitted curves."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    dates = pd.to_datetime(indicators_df['测量日期'])
    t_days = np.array(ode_results['t_days'])

    # Panel 1: Area A(t) with logistic fit
    ax = axes[0]
    A_true = indicators_df['A_面积(m²)'].values
    A_pred = ode_results['A']['predicted']
    A_eq = ode_results['A']['A_eq']

    ax.scatter(dates, A_true, c='steelblue', s=60, zorder=5, label='Measured A(t)')
    ax.plot(dates, A_pred, 'r-', linewidth=2, alpha=0.7, label='Logistic fit')
    ax.axhline(y=A_eq, color='green', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'$A_{{eq}}$ = {A_eq:.0f} m²')

    ax.set_ylabel('Cross-sectional Area A (m²)', fontsize=11)
    ax.set_title(f'A(t): Logistic Growth Model (R²={ode_results["A"]["r2"]:.3f}, '
                 f'Stable={ode_results["A"]["is_stable"]})', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30)

    # Panel 2: Width-depth ratio ξ(t)
    ax = axes[1]
    xi_true = indicators_df['ξ_宽深比'].values
    xi_pred = ode_results['ξ']['predicted']
    xi_eq = ode_results['ξ']['X_eq']

    ax.scatter(dates, xi_true, c='darkorange', s=60, zorder=5, marker='s',
               label='Measured ξ(t)')
    ax.plot(dates, xi_pred, 'r-', linewidth=2, alpha=0.7, label='Exponential fit')
    ax.axhline(y=xi_eq, color='green', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'$ξ_{{eq}}$ = {xi_eq:.2f}')

    ax.set_ylabel('Width-depth Ratio ξ = B²/A', fontsize=11)
    ax.set_title(f'ξ(t): Exponential Approach (R²={ode_results["ξ"]["r2"]:.3f})',
                 fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30)

    # Panel 3: Morphological entropy H(t)
    ax = axes[2]
    H_true = indicators_df['H_形态熵(nats)'].values
    H_pred = ode_results['H']['predicted']
    H_eq = ode_results['H']['X_eq']

    ax.scatter(dates, H_true, c='seagreen', s=60, zorder=5, marker='^',
               label='Measured H(t)')
    ax.plot(dates, H_pred, 'r-', linewidth=2, alpha=0.7, label='Exponential fit')
    ax.axhline(y=H_eq, color='green', linestyle='--', linewidth=1.5,
               alpha=0.6, label=f'$H_{{eq}}$ = {H_eq:.4f} nats')

    ax.set_ylabel('Morphological Entropy H (nats)', fontsize=11)
    ax.set_xlabel('Measurement Date', fontsize=12)
    ax.set_title(f'H(t): Exponential Approach (R²={ode_results["H"]["r2"]:.3f})',
                 fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved evolution plot → {output_path}")


def plot_loocv(predictions: dict, output_path: str):
    """Plot LOOCV predicted vs actual."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ΔA
    ax = axes[0]
    y_true = np.array(predictions['ΔA_true'])
    y_pred = np.array(predictions['ΔA_pred'])

    ax.scatter(y_true, y_pred, c='steelblue', s=80, zorder=5)
    lims = [min(y_true.min(), y_pred.min()) - 50, max(y_true.max(), y_pred.max()) + 50]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect prediction')
    ax.set_xlabel('True ΔA (m²)', fontsize=11)
    ax.set_ylabel('Predicted ΔA (m²)', fontsize=11)
    ax.set_title('LOOCV: ΔA Prediction', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Δξ
    ax = axes[1]
    y_true = np.array(predictions['Δξ_true'])
    y_pred = np.array(predictions['Δξ_pred'])

    ax.scatter(y_true, y_pred, c='darkorange', s=80, zorder=5, marker='s')
    lims = [min(y_true.min(), y_pred.min()) - 10, max(y_true.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect prediction')
    ax.set_xlabel('True Δξ', fontsize=11)
    ax.set_ylabel('Predicted Δξ', fontsize=11)
    ax.set_title('LOOCV: Δξ Prediction', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved LOOCV plot → {output_path}")


def main():
    from preprocess import load_cross_sections, process_all_sections
    from geometry import calculate_all_indicators
    from ode_model import fit_all_steady_state

    data_dir = Path(__file__).parent.parent / 'data_csv'
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)

    # Load data
    sections_file = data_dir / '附件2_9个断面地形数据.csv'
    raw_sections = load_cross_sections(str(sections_file))
    sections = process_all_sections(raw_sections, dx=1.0)

    indicators_df = calculate_all_indicators(sections)
    ode_results = fit_all_steady_state(indicators_df)

    # Plot 1: Cross-sections
    print("\n--- Figure 1: Cross-Sections ---")
    plot_cross_sections(
        sections,
        str(results_dir / 'figure_cross_sections.png')
    )

    # Plot 2: Evolution curves
    print("\n--- Figure 2: Evolution Curves ---")
    plot_evolution(
        indicators_df, ode_results,
        str(results_dir / 'figure_evolution.png')
    )

    # Plot 3: LOOCV (needs full pipeline)
    print("\n--- Figure 3: LOOCV ---")
    from loocv import main as loocv_main
    metrics_A, metrics_xi, predictions = loocv_main()
    plot_loocv(
        predictions,
        str(results_dir / 'figure_loocv.png')
    )

    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
