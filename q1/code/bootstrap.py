"""
Module: Bootstrap Validation (v4)
B = 1000 bootstrap resamples to estimate R^2 distribution and 95% CI.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = 1000,
    seed: int = 42
) -> dict:
    """
    Bootstrap R^2 estimation.

    Returns:
        dict with mean_r2, ci_95, all_r2 values, std_r2
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    r2_samples = np.zeros(B)

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        r2_samples[b] = compute_r2(y_true[idx], y_pred[idx])

    mean_r2 = np.mean(r2_samples)
    ci_low = np.percentile(r2_samples, 2.5)
    ci_high = np.percentile(r2_samples, 97.5)
    std_r2 = np.std(r2_samples)

    print(f"  Bootstrap R^2 (B={B}): mean={mean_r2:+.4f}, "
          f"95% CI=[{ci_low:+.4f}, {ci_high:+.4f}], std={std_r2:.4f}")

    return {
        'r2_boot_mean': round(mean_r2, 6),
        'r2_boot_std': round(std_r2, 6),
        'r2_boot_ci_low': round(ci_low, 6),
        'r2_boot_ci_high': round(ci_high, 6),
        'r2_samples': r2_samples.tolist(),
        'B': B
    }


def bootstrap_all_targets(
    ensemble_results: dict,
    B: int = 1000
) -> dict:
    """
    Run bootstrap on all targets' ensemble predictions (LOOCV).

    Args:
        ensemble_results: {target_name: {'ensemble_result': ..., 'base_results': ...}}

    Returns:
        dict: {target_name: bootstrap_result}
    """
    print(f"\n{'=' * 60}")
    print(f"Bootstrap Validation (B={B})")
    print(f"{'=' * 60}")

    boot_results = {}

    for target_name, er in ensemble_results.items():
        best_config = er['ensemble_result']['best_config']
        y_pred_loo = best_config.get('y_pred_loo', None)

        if y_pred_loo is None:
            # Use best single model
            base = er['base_results']
            best_name = max(base.items(),
                            key=lambda kv: kv[1]['r2_loo'])[0]
            y_pred_loo = base[best_name]['y_pred_loo']

        # Get y_true from any base model
        y_true = np.array(list(er['base_results'].values())[0]['y_true'])

        print(f"\n  [{target_name}]")
        boot = bootstrap_r2(y_true, np.array(y_pred_loo), B=B)
        boot_results[target_name] = boot

        # Also bootstrap per base model
        for model_name, model_res in er['base_results'].items():
            if not model_res.get('active', False):
                continue
            boot_m = bootstrap_r2(
                np.array(model_res['y_true']),
                np.array(model_res['y_pred_loo']),
                B=B
            )
            boot_results[f'{target_name}_{model_name}'] = boot_m

    return boot_results


def save_bootstrap_results(boot_results: dict, output_path: str):
    """Save bootstrap summary to CSV."""
    rows = []
    for name, res in boot_results.items():
        rows.append({
            'target_model': name,
            'R2_Boot_mean': res['r2_boot_mean'],
            'R2_Boot_std': res['r2_boot_std'],
            'R2_Boot_CI_low': res['r2_boot_ci_low'],
            'R2_Boot_CI_high': res['r2_boot_ci_high'],
            'B': res['B']
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved bootstrap results to {output_path}")

    return df


if __name__ == '__main__':
    print("Bootstrap module loaded. Use run_all.py for full pipeline.")
