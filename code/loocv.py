"""
Module: LOOCV (v4) — Leave-One-Out Cross-Validation
Adapted for the BMA ensemble framework.
Provides LOOCV wrappers for base models and ensemble predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_loo_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard LOOCV metrics."""
    errors = y_pred - y_true
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    nonzero = np.abs(y_true) > 1e-10
    mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100) \
           if nonzero.sum() > 0 else np.nan

    return {'RMSE': round(rmse, 4), 'R2': round(r2, 4), 'MAPE(%)': round(mape, 2),
            'n': len(y_true)}


def save_loo_results(
    ensemble_results: dict,
    output_path: str
):
    """Save LOOCV results from ensemble pipeline to CSV."""
    rows = []

    for target_name, er in ensemble_results.items():
        # Best single model
        base = er['base_results']
        best_single = max(base.items(), key=lambda kv: kv[1]['r2_loo'])
        best_name = best_single[0]
        best_res = best_single[1]

        # Best ensemble
        best_config = er['ensemble_result']['best_config']
        strategy = er['ensemble_result']['best_strategy']
        ensemble_r2 = best_config.get('_r2', np.nan)

        y_true = np.array(best_res['y_true'])
        y_pred_ens = np.array(best_config.get('y_pred_loo', best_res['y_pred_loo']))

        metrics_ens = compute_loo_metrics(y_true, y_pred_ens)
        metrics_single = compute_loo_metrics(
            y_true, np.array(best_res['y_pred_loo']))

        rows.append({
            '指标': target_name,
            '最佳单模型': best_name,
            '单模型R2_LOO': round(best_res['r2_loo'], 4),
            '单模型RMSE': metrics_single['RMSE'],
            '集成策略': strategy,
            '集成R2_LOO': round(ensemble_r2, 4),
            '集成RMSE': metrics_ens['RMSE'],
            '集成MAPE(%)': metrics_ens['MAPE(%)'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved LOOCV summary to {output_path}")
    print(df.to_string(index=False))

    return df


def main():
    print("LOOCV module — use run_all.py for full pipeline.")
