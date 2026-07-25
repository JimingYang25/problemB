"""
Main runner script — executes the full pipeline (Modules 1-6 + figures).
Run: python code/run_all.py
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from preprocess import load_cross_sections, process_all_sections, save_processed_data
from geometry import calculate_all_indicators, calculate_deltas
from driving_factors import load_hydro_data, extract_driving_factors
from regression import build_regression_data, fit_both_equations, save_regression_results
from ode_model import fit_all_steady_state, save_steady_state_results
from loocv import loocv_validation, save_validation_results

DATA_DIR = Path(__file__).parent.parent / 'data_csv'
DATA_OUT = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Ensure output directories exist
DATA_OUT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    print("=" * 70)
    print("  黄河水沙通量变化规律 — 第一问完整建模流水线")
    print("  Yellow River Water-Sediment Flux — Question 1 Full Pipeline")
    print("=" * 70)

    # =========================================================================
    # Module 1: Data Preprocessing
    # =========================================================================
    print("\n" + "█" * 60)
    print("█  Module 1: Data Preprocessing")
    print("█" * 60)
    sections_file = DATA_DIR / '附件2_9个断面地形数据.csv'
    raw_sections = load_cross_sections(str(sections_file))
    sections = process_all_sections(raw_sections, dx=1.0)
    save_processed_data(sections, str(DATA_OUT / 'cross_sections_processed.csv'))

    # =========================================================================
    # Module 2: Geometric Indicators
    # =========================================================================
    print("\n" + "█" * 60)
    print("█  Module 2: Geometric Indicators (A, B, ξ, H)")
    print("█" * 60)
    indicators_df = calculate_all_indicators(sections)
    deltas_df = calculate_deltas(indicators_df)
    indicators_df.to_csv(DATA_OUT / 'geometry_indicators.csv',
                         index=False, encoding='utf-8-sig')
    deltas_df.to_csv(DATA_OUT / 'geometry_deltas.csv',
                     index=False, encoding='utf-8-sig')
    print(f"  Saved: geometry_indicators.csv, geometry_deltas.csv")

    # =========================================================================
    # Module 3: Driving Factors
    # =========================================================================
    print("\n" + "█" * 60)
    print("█  Module 3: Driving Factors (V, M, Qpeak)")
    print("█" * 60)
    hydro_file = DATA_DIR / '附件1_逐小时水沙数据_2016-2021.csv'
    hydro_df = load_hydro_data(str(hydro_file))
    section_dates = sorted(sections.keys())
    date_pairs = [(section_dates[i], section_dates[i + 1])
                  for i in range(len(section_dates) - 1)]
    factors_df = extract_driving_factors(hydro_df, date_pairs, section_dates)
    factors_df.to_csv(DATA_OUT / 'driving_factors.csv',
                      index=False, encoding='utf-8-sig')
    print(f"  Saved: driving_factors.csv")

    # =========================================================================
    # Module 4: Empirical Formula Fitting (Regression)
    # =========================================================================
    print("\n" + "█" * 60)
    print("█  Module 4: Empirical Formula Fitting")
    print("█" * 60)
    merged = build_regression_data(deltas_df, factors_df)
    result_A, result_xi = fit_both_equations(merged)
    save_regression_results(
        result_A, result_xi,
        str(RESULTS_DIR / 'regression_coefficients.csv')
    )

    # =========================================================================
    # Module 5: Steady-State Equation (Logistic ODE)
    # =========================================================================
    print("\n" + "█" * 60)
    print("█  Module 5: Steady-State Equation Fitting")
    print("█" * 60)
    ode_results = fit_all_steady_state(indicators_df)
    save_steady_state_results(
        ode_results,
        str(RESULTS_DIR / 'steady_state_params.csv')
    )

    # =========================================================================
    # Module 6: LOOCV Validation
    # =========================================================================
    print("\n" + "█" * 60)
    print("█  Module 6: LOOCV Validation")
    print("█" * 60)
    metrics_A, metrics_xi, predictions = loocv_validation(merged, n_sections=9)
    save_validation_results(
        metrics_A, metrics_xi, predictions,
        str(RESULTS_DIR / 'validation_metrics.csv')
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
  Output Files:
    data/
      ├── cross_sections_processed.csv
      ├── geometry_indicators.csv
      ├── geometry_deltas.csv
      └── driving_factors.csv

    results/
      ├── regression_coefficients.csv
      ├── steady_state_params.csv
      ├── validation_metrics.csv
      ├── validation_metrics_per_fold.csv
      ├── figure_cross_sections.png
      ├── figure_evolution.png
      └── figure_loocv.png

  Key Findings:
    Regression:
      ΔA R² = {result_A['r2']:.4f} (RMSE={result_A['rmse']:.1f})
      Δξ R² = {result_xi['r2']:.4f} (RMSE={result_xi['rmse']:.4f})

    Steady State:
      A_eq = {ode_results['A']['A_eq']:.1f} m² (stable={ode_results['A']['is_stable']})

    LOOCV:
      ΔA: RMSE={metrics_A['RMSE']:.2f}, MAPE={metrics_A['MAPE(%)']:.2f}%
      Δξ: RMSE={metrics_xi['RMSE']:.4f}, MAPE={metrics_xi['MAPE(%)']:.2f}%
""")

    return {
        'indicators': indicators_df,
        'deltas': deltas_df,
        'factors': factors_df,
        'merged': merged,
        'regression_A': result_A,
        'regression_xi': result_xi,
        'ode': ode_results,
        'loocv_metrics_A': metrics_A,
        'loocv_metrics_xi': metrics_xi,
        'loocv_predictions': predictions,
    }


if __name__ == '__main__':
    results = run_pipeline()
