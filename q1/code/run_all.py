"""
Main runner — v4/v5 BMA Pipeline
Hybrid DRM + Adaptive Bayesian Model Averaging

Run: cd g:/problemB && PYTHONIOENCODING=utf-8 python code/run_all.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from q1.code.preprocess import load_cross_sections, process_all_sections, save_processed_data
from q1.code.geometry import calculate_all_indicators, calculate_deltas, Z_REF
from q1.code.driving_factors import load_hydro_data, reconstruct_sediment, to_daily
from q1.code.driving_factors import extract_features, select_features_for_target
from q1.code.drm_model import fit_drm_all_targets
from q1.code.ensemble import run_ensemble_for_target, run_ensemble_all_targets
from q1.code.bootstrap import bootstrap_r2, bootstrap_all_targets, save_bootstrap_results

DATA_DIR = Path(__file__).parent.parent / 'data_csv'
DATA_OUT = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

DATA_OUT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    print("=" * 70)
    print("  BMA v4/v5 Pipeline: DRM + Adaptive Ensemble")
    print("  Yellow River Cross-Section Morphology Prediction")
    print("=" * 70)

    # =========================================================================
    # Step 1: Data Preprocessing
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 1: Cross-Section Preprocessing")
    print("#" * 60)
    sections_file = DATA_DIR / '附件2_9个断面地形数据.csv'
    raw_sections = load_cross_sections(str(sections_file))
    sections = process_all_sections(raw_sections, dx=1.0)
    save_processed_data(sections, str(DATA_OUT / 'cross_sections_processed.csv'))
    section_dates = sorted(sections.keys())

    # =========================================================================
    # Step 2: Geometric Indicators (5 targets)
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 2: Geometric Indicators (A, B, xi, H, z_min)")
    print("#" * 60)
    indicators_df = calculate_all_indicators(sections)
    deltas_df = calculate_deltas(indicators_df)

    indicators_df.to_csv(DATA_OUT / 'geometry_indicators.csv',
                         index=False, encoding='utf-8-sig')
    deltas_df.to_csv(DATA_OUT / 'geometry_deltas.csv',
                     index=False, encoding='utf-8-sig')

    print("\n  Indicators summary:")
    print(indicators_df.to_string(index=False))
    print("\n  Deltas summary:")
    print(deltas_df.to_string(index=False))

    # =========================================================================
    # Step 3: Feature Engineering
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 3: Feature Engineering (14-dim + S-Q reconstruction)")
    print("#" * 60)

    # Load hydro data
    hydro_file = DATA_DIR / '附件1_逐小时水沙数据_2016-2021.csv'
    hydro_df = load_hydro_data(str(hydro_file))

    # S-Q reconstruction
    print("\n--- Sediment Reconstruction ---")
    hydro_df, rating = reconstruct_sediment(hydro_df)

    # Daily aggregation
    daily = to_daily(hydro_df)
    print(f"  Daily: {len(daily)} days, Q=[{daily['Q'].min():.0f}, {daily['Q'].max():.0f}], "
          f"S=[{daily['S'].min():.3f}, {daily['S'].max():.1f}]")

    # Build date pairs
    date_pairs = [(section_dates[i], section_dates[i + 1])
                  for i in range(len(section_dates) - 1)]

    # Extract features
    print("\n--- Feature Extraction ---")
    features = extract_features(daily, date_pairs)
    features.to_csv(DATA_OUT / 'driving_factors.csv', index=False, encoding='utf-8-sig')

    # Add Qavg and Savg for DRM
    features['Qavg'] = features['V'].values / (features['T_days'].values * 86400)
    features['Savg'] = features['Qs'].values / (features['Qavg'].values + 1e-10)

    # =========================================================================
    # Step 4: Feature Selection per Target
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 4: Feature Selection (Spearman |rho| + interaction)")
    print("#" * 60)

    FEATURE_SELECTION = {}
    for col, short in [('dA', 'dA'), ('dB', 'dB'), ('dxi', 'dxi'),
                        ('dH', 'dH'), ('dz_min', 'dz_min')]:
        if col in deltas_df.columns:
            X_sel, names = select_features_for_target(
                features, deltas_df[col].values, short
            )
            FEATURE_SELECTION[short] = names

    # =========================================================================
    # Step 5: DRM Model Fitting
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 5: DRM (Delayed Response Model) Fitting")
    print("#" * 60)
    drm_results = fit_drm_all_targets(features, deltas_df, indicators_df)

    # Save DRM results
    drm_rows = []
    for name, r in drm_results.items():
        drm_rows.append({
            '指标': name,
            'K': round(r['K'], 6),
            'a': round(r['a'], 6),
            'b': round(r['b'], 6),
            'beta': round(r['beta'], 6),
            'R2': round(r['r2'], 4),
            'RMSE': round(r['rmse'], 4),
            'Stable?': r['is_stable'],
            'tau(periods)': round(r['tau'], 2),
            'memory_factor': round(r['memory_factor'], 4)
        })
    pd.DataFrame(drm_rows).to_csv(
        RESULTS_DIR / 'drm_parameters.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved DRM parameters to drm_parameters.csv")

    # =========================================================================
    # Step 6: BMA Ensemble
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 6: BMA Ensemble (7 models x 6 strategies)")
    print("#" * 60)

    ensemble_results, summary_df = run_ensemble_all_targets(
        features, deltas_df, FEATURE_SELECTION
    )

    summary_df.to_csv(
        RESULTS_DIR / 'ensemble_summary.csv', index=False, encoding='utf-8-sig')

    # Save ensemble weights
    weight_rows = []
    for target_name, er in ensemble_results.items():
        best = er['ensemble_result']['best_config']
        strategy = er['ensemble_result']['best_strategy']
        for model, w in best.get('weights', {}).items():
            weight_rows.append({
                '指标': target_name,
                '策略': strategy,
                '模型': model,
                '权重': w
            })
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(
            RESULTS_DIR / 'ensemble_weights.csv', index=False, encoding='utf-8-sig')

    # =========================================================================
    # Step 7: Bootstrap Validation
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 7: Bootstrap Validation (B=1000)")
    print("#" * 60)

    boot_results = bootstrap_all_targets(ensemble_results, B=1000)
    save_bootstrap_results(boot_results, str(RESULTS_DIR / 'bootstrap_summary.csv'))

    # =========================================================================
    # Step 8: Steady-State Analysis
    # =========================================================================
    print("\n" + "#" * 60)
    print("#  STEP 8: Steady-State Analysis")
    print("#" * 60)
    steady_state_analysis(drm_results, indicators_df)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — v4/v5 BMA + DRM")
    print("=" * 70)
    print(f"""
  Output Files:
    data/
      cross_sections_processed.csv
      geometry_indicators.csv
      geometry_deltas.csv
      driving_factors.csv
    results/
      drm_parameters.csv          — DRM (K, a, b, beta, R^2, stability)
      ensemble_summary.csv        — Best single model vs ensemble per target
      ensemble_weights.csv        — Optimal ensemble weights
      bootstrap_summary.csv       — Bootstrap R^2 distributions
      steady_state_analysis.csv   — Equilibrium values & convergence

  Key Architecture:
    7 base models:  Ridge, GPR, KRR, ElasticNet, Lasso, Huber, DRM
    6 strategies:   Best-Single, Equal, BMA-BIC, Softmax-LOO,
                    Softmax-Boot, Stacking(NNLS)
    Adaptive:       Model filtering (R^2 > -0.3) + strategy selection
    Validation:     LOOCV + Bootstrap (B=1000)
""")
    return ensemble_results, drm_results, boot_results


def steady_state_analysis(drm_results: dict, indicators_df: pd.DataFrame):
    """Analyze steady-state convergence for all DRM-fitted indicators."""
    rows = []
    for name, r in drm_results.items():
        y_obs = r['y_obs']
        y_hat = r['y_hat']
        y_e_last = r['y_e'][-1] if r['y_e'] else np.nan
        current = y_obs[-1]
        delta = y_e_last - current
        delta_pct = 100 * delta / abs(current) if abs(current) > 1e-10 else 0

        rows.append({
            '指标': name,
            '当前值': round(current, 4),
            '平衡值 y_eq': round(y_e_last, 4),
            '偏离量': round(delta, 4),
            '偏离比例(%)': round(delta_pct, 1),
            'beta': round(r['beta'], 4),
            '松弛时间 tau': round(r['tau'], 2),
            '记忆因子': round(r['memory_factor'], 4),
            'R^2': round(r['r2'], 4),
            '稳定?': r['is_stable']
        })

        status = "STABLE" if r['is_stable'] else "UNSTABLE"
        print(f"  [{name}] current={current:.2f}, y_eq={y_e_last:.2f}, "
              f"delta={delta:+.2f} ({delta_pct:+.1f}%), "
              f"beta={r['beta']:.3f}, tau={r['tau']:.1f}, {status}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'steady_state_analysis.csv',
              index=False, encoding='utf-8-sig')
    print(f"\nSaved steady-state analysis to steady_state_analysis.csv")


if __name__ == '__main__':
    run_pipeline()
