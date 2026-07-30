"""
Module: BMA Ensemble (v4/v5)
7 base models + 6 ensemble strategies + adaptive model selection

Base models: Ridge, GPR, KRR, ElasticNet, Lasso, Huber, DRM
Ensemble strategies: Best-Single, Equal, BMA-BIC, Softmax-LOO,
                     Softmax-Boot, Stacking(NNLS)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import (
    Ridge, ElasticNet, Lasso, HuberRegressor, LinearRegression
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls


# ============================================================================
# Base Model Training
# ============================================================================

def train_ridge(X, y, alpha=1.0):
    """Ridge regression (L2 regularized)."""
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, y)
    return m

def train_gpr(X, y, alpha=0.1):
    """Gaussian Process Regression with RBF + White kernel."""
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=alpha)
    m = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                  normalize_y=True, random_state=42)
    m.fit(X, y)
    return m

def train_krr(X, y, alpha=1.0, gamma=0.1):
    """Kernel Ridge Regression with RBF kernel."""
    m = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    m.fit(X, y)
    return m

def train_elasticnet(X, y, alpha=0.1, l1_ratio=0.5):
    """ElasticNet (L1 + L2 mixed)."""
    m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                   max_iter=10000, random_state=42)
    m.fit(X, y)
    return m

def train_lasso(X, y, alpha=0.1):
    """Lasso (L1 regularized)."""
    m = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000, random_state=42)
    m.fit(X, y)
    return m

def train_huber(X, y, epsilon=1.35, alpha=0.001):
    """Huber robust regression."""
    m = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000,
                       fit_intercept=True)
    m.fit(X, y)
    return m


# ============================================================================
# LOOCV for base models
# ============================================================================

def loocv_single_model(train_fn, X, y):
    """
    Leave-one-out CV for a single model.
    train_fn(X_train, y_train) → model with .predict() method.
    """
    n = len(X)
    y_pred = np.zeros(n)
    y_true = np.zeros(n)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[i:i + 1]

        try:
            model = train_fn(X_train, y_train)
            y_pred[i] = model.predict(X_test)[0]
        except Exception:
            y_pred[i] = np.mean(y_train)

        y_true[i] = y[i]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return {
        'r2': r2, 'rmse': rmse,
        'y_pred': y_pred.tolist(),
        'y_true': y_true.tolist()
    }


def loocv_drm_wrapper(drm_loocv_fn, X, y):
    """
    LOOCV for DRM (uses entire time series structure).
    drm_loocv_fn(train_idx_list, X, y) → dict with 'y_pred_loo' list
    """
    return drm_loocv_fn(X, y)


# ============================================================================
# Hyperparameter Loops via LOOCV
# ============================================================================

def best_ridge(X, y):
    """Select best alpha for Ridge via LOOCV."""
    best_r2, best_model, best_alpha = -np.inf, None, 0.1
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        loo = loocv_single_model(lambda Xt, yt: train_ridge(Xt, yt, alpha), X, y)
        if loo['r2'] > best_r2:
            best_r2, best_model, best_alpha = loo['r2'], loo, alpha
    result = train_ridge(X, y, best_alpha)  # fit on full data
    return result, best_r2, best_alpha


def best_gpr(X, y):
    """Select best alpha for GPR via LOOCV."""
    best_r2, best_alpha = -np.inf, 0.1
    for alpha in [0.05, 0.1, 0.3]:
        loo = loocv_single_model(lambda Xt, yt: train_gpr(Xt, yt, alpha), X, y)
        if loo['r2'] > best_r2:
            best_r2, best_alpha = loo['r2'], alpha
    result = train_gpr(X, y, best_alpha)
    return result, best_r2, best_alpha


def best_krr(X, y):
    """Select best (alpha, gamma) for KRR via LOOCV."""
    best_r2, best_params = -np.inf, (1.0, 0.1)
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
        for gamma in [0.01, 0.1, 0.5, 1.0]:
            loo = loocv_single_model(
                lambda Xt, yt: train_krr(Xt, yt, alpha, gamma), X, y)
            if loo['r2'] > best_r2:
                best_r2, best_params = loo['r2'], (alpha, gamma)
    result = train_krr(X, y, *best_params)
    return result, best_r2, best_params


def best_elasticnet(X, y):
    """Select best (alpha, l1_ratio) for ElasticNet via LOOCV."""
    best_r2, best_params = -np.inf, (0.1, 0.5)
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            loo = loocv_single_model(
                lambda Xt, yt: train_elasticnet(Xt, yt, alpha, l1), X, y)
            if loo['r2'] > best_r2:
                best_r2, best_params = loo['r2'], (alpha, l1)
    result = train_elasticnet(X, y, *best_params)
    return result, best_r2, best_params


def best_lasso(X, y):
    """Select best alpha for Lasso via LOOCV."""
    best_r2, best_alpha = -np.inf, 0.1
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        loo = loocv_single_model(lambda Xt, yt: train_lasso(Xt, yt, alpha), X, y)
        if loo['r2'] > best_r2:
            best_r2, best_alpha = loo['r2'], alpha
    result = train_lasso(X, y, best_alpha)
    return result, best_r2, best_alpha


def best_huber(X, y):
    """Select best (epsilon, alpha) for Huber via LOOCV."""
    best_r2, best_params = -np.inf, (1.35, 0.001)
    for eps in [1.1, 1.35, 1.5, 2.0]:
        for alpha in [0.0001, 0.001, 0.01, 0.1]:
            loo = loocv_single_model(
                lambda Xt, yt: train_huber(Xt, yt, eps, alpha), X, y)
            if loo['r2'] > best_r2:
                best_r2, best_params = loo['r2'], (eps, alpha)
    result = train_huber(X, y, *best_params)
    return result, best_r2, best_params


# ============================================================================
# Full Base Model Training with LOOCV
# ============================================================================

MODEL_FACTORIES = {
    'Ridge': best_ridge,
    'GPR': best_gpr,
    'KRR': best_krr,
    'ElasticNet': best_elasticnet,
    'Lasso': best_lasso,
    'Huber': best_huber,
}


def train_all_base_models(X, y, active_models=None) -> dict:
    """
    Train all base models with LOOCV hyperparameter selection.

    Returns:
        dict: {model_name: {'model': fitted_model, 'r2_loo': float,
                            'rmse_loo': float, 'y_pred_loo': list, ...}}
    """
    if active_models is None:
        active_models = list(MODEL_FACTORIES.keys())

    results = {}
    X_scaled = StandardScaler().fit_transform(X)  # standardize for regularized models

    print(f"\n  Training base models (n={len(X)})...")

    for name in active_models:
        if name not in MODEL_FACTORIES:
            continue
        try:
            model, r2, params = MODEL_FACTORIES[name](X_scaled, y)
            # Re-run LOOCV to get predictions
            loo = loocv_single_model(
                lambda Xt, yt: train_single(name, Xt, yt, params), X_scaled, y)

            active = r2 > -0.3  # BMA screening threshold
            results[name] = {
                'model': model,
                'r2_loo': round(r2, 4),
                'rmse_loo': round(loo['rmse'], 4),
                'y_pred_loo': loo['y_pred'],
                'y_true': loo['y_true'],
                'params': params,
                'active': active,
                'n_params': 2 if isinstance(params, tuple) else 1
            }
            status = "ACTIVE" if active else "dropped"
            print(f"    {name:12s}: R^2_LOO={r2:+.4f}, RMSE={loo['rmse']:.4f}  [{status}]")
        except Exception as e:
            print(f"    {name:12s}: FAILED — {e}")
            results[name] = {
                'model': None, 'r2_loo': -np.inf, 'active': False,
                'y_pred_loo': np.zeros(len(y)).tolist(),
                'y_true': y.tolist()
            }

    return results


def train_single(name, X, y, params):
    """Train a single model given name and params (from best_* functions)."""
    if name == 'Ridge':
        return train_ridge(X, y, alpha=params)
    elif name == 'GPR':
        return train_gpr(X, y, alpha=params)
    elif name == 'KRR':
        return train_krr(X, y, alpha=params[0], gamma=params[1])
    elif name == 'ElasticNet':
        return train_elasticnet(X, y, alpha=params[0], l1_ratio=params[1])
    elif name == 'Lasso':
        return train_lasso(X, y, alpha=params)
    elif name == 'Huber':
        return train_huber(X, y, epsilon=params[0], alpha=params[1])
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================================
# Ensemble Strategies
# ============================================================================

def ensemble_best_single(base_results: dict) -> dict:
    """Select the single best model by LOOCV R^2."""
    best_name = max(
        [k for k, v in base_results.items() if v['active']],
        key=lambda k: base_results[k]['r2_loo']
    )
    return {
        'strategy': 'Best-Single',
        'weights': {best_name: 1.0},
        'model_name': best_name
    }


def ensemble_equal(base_results: dict) -> dict:
    """Equal-weight all active models."""
    active = {k: v for k, v in base_results.items() if v['active']}
    w = 1.0 / len(active) if active else 0
    return {
        'strategy': 'Equal',
        'weights': {k: w for k in active}
    }


def ensemble_bma_bic(base_results: dict, n: int) -> dict:
    """BIC-weighted ensemble."""
    active = {k: v for k, v in base_results.items() if v['active']}
    if not active:
        return {'strategy': 'BMA-BIC', 'weights': {}}

    bics = {}
    for name, res in active.items():
        y_true = np.array(res['y_true'])
        y_pred = np.array(res['y_pred_loo'])
        sse = np.sum((y_true - y_pred) ** 2)
        k = res.get('n_params', 2)
        bics[name] = n * np.log(sse / n + 1e-10) + k * np.log(n)

    min_bic = min(bics.values())
    raw = {k: np.exp(-0.5 * (v - min_bic)) for k, v in bics.items()}
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()} if total > 0 else {}

    return {'strategy': 'BMA-BIC', 'weights': weights, 'bics': bics}


def ensemble_softmax_loo(base_results: dict, T: float = 0.1) -> dict:
    """Softmax-weighted by LOOCV R^2 with temperature T."""
    active = {k: v for k, v in base_results.items() if v['active']}
    if not active:
        return {'strategy': f'Softmax-LOO(T={T})', 'weights': {}}

    r2s = {k: v['r2_loo'] for k, v in active.items()}
    max_r2 = max(r2s.values()) if r2s else 0
    raw = {k: np.exp((v - max_r2) / T) for k, v in r2s.items()}
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()} if total > 0 else {}

    return {
        'strategy': f'Softmax-LOO(T={T})',
        'weights': weights,
        'temperature': T
    }


def ensemble_softmax_boot(base_results: dict, boot_results: dict = None, T: float = 0.1) -> dict:
    """
    Softmax-weighted by composite score:
      S_m = 0.5 * max(R^2_LOO, -1) + 0.5 * max(R^2_Boot, -1)
    """
    active = {k: v for k, v in base_results.items() if v['active']}
    if not active:
        return {'strategy': f'Softmax-Boot(T={T})', 'weights': {}}

    scores = {}
    for name in active:
        r2_loo = base_results[name]['r2_loo']
        r2_boot = boot_results.get(name, {}).get('r2_boot_mean', r2_loo) \
                  if boot_results else r2_loo
        scores[name] = 0.5 * max(r2_loo, -1.0) + 0.5 * max(r2_boot, -1.0)

    max_s = max(scores.values()) if scores else 0
    raw = {k: np.exp((v - max_s) / T) for k, v in scores.items()}
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()} if total > 0 else {}

    return {
        'strategy': f'Softmax-Boot(T={T})',
        'weights': weights,
        'scores': scores,
        'temperature': T
    }


def ensemble_stacking(base_results: dict) -> dict:
    """
    Non-Negative Least Squares stacking on LOOCV predictions.
    min_{w >= 0} || y - P_LOO * w ||^2
    """
    active = {k: v for k, v in base_results.items() if v['active']}
    if not active:
        return {'strategy': 'Stacking(NNLS)', 'weights': {}}

    y_true = np.array(list(active.values())[0]['y_true'])
    P = np.column_stack([v['y_pred_loo'] for v in active.values()])

    w, residual = nnls(P, y_true)
    w = w / w.sum() if w.sum() > 0 else w

    weights = {name: float(w[i]) for i, name in enumerate(active.keys())}

    return {
        'strategy': 'Stacking(NNLS)',
        'weights': weights,
        'stacking_r2': float(1 - np.sum(residual ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    }


# ============================================================================
# Adaptive Strategy Selection
# ============================================================================

def evaluate_strategy(strategy_result: dict, base_results: dict) -> float:
    """
    Compute LOOCV R^2 for a given ensemble strategy.
    Uses weighted average of base model LOOCV predictions.
    """
    weights = strategy_result.get('weights', {})
    if not weights:
        return -np.inf

    y_true = None
    y_pred = np.zeros(len(list(base_results.values())[0]['y_true']))

    for name, w in weights.items():
        if name not in base_results or not base_results[name]['active']:
            continue
        if y_true is None:
            y_true = np.array(base_results[name]['y_true'])
        y_pred += w * np.array(base_results[name]['y_pred_loo'])

    if y_true is None:
        return -np.inf

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def adaptive_ensemble(base_results: dict, boot_results: dict = None) -> dict:
    """
    Evaluate all 6 strategies and select the best by LOOCV R^2.
    """
    n = len(list(base_results.values())[0]['y_true'])
    temperatures = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    strategies = {}

    # Best-Single
    strategies['Best-Single'] = ensemble_best_single(base_results)

    # Equal
    strategies['Equal'] = ensemble_equal(base_results)

    # BMA-BIC
    strategies['BMA-BIC'] = ensemble_bma_bic(base_results, n)

    # Softmax-LOO (best T)
    best_softmax_loo = None
    best_r2_loo = -np.inf
    for T in temperatures:
        s = ensemble_softmax_loo(base_results, T)
        r2 = evaluate_strategy(s, base_results)
        s['_r2'] = r2
        if r2 > best_r2_loo:
            best_r2_loo = r2
            best_softmax_loo = s
    strategies['Softmax-LOO'] = best_softmax_loo

    # Softmax-Boot (best T)
    best_softmax_boot = None
    best_r2_boot = -np.inf
    for T in temperatures:
        s = ensemble_softmax_boot(base_results, boot_results, T)
        r2 = evaluate_strategy(s, base_results)
        s['_r2'] = r2
        if r2 > best_r2_boot:
            best_r2_boot = r2
            best_softmax_boot = s
    strategies['Softmax-Boot'] = best_softmax_boot

    # Stacking
    strategies['Stacking'] = ensemble_stacking(base_results)
    strategies['Stacking']['_r2'] = evaluate_strategy(strategies['Stacking'], base_results)

    # Select best
    best_strategy = max(strategies.items(),
                        key=lambda kv: kv[1].get('_r2', -np.inf))

    # Merge: add y_pred_loo for best strategy
    best_name, best_config = best_strategy
    y_pred = np.zeros(len(list(base_results.values())[0]['y_true']))
    for name, w in best_config.get('weights', {}).items():
        if name in base_results:
            y_pred += w * np.array(base_results[name]['y_pred_loo'])
    best_config['y_pred_loo'] = y_pred.tolist()

    print(f"\n  --- Ensemble Strategy Rankings ---")
    for name in ['Best-Single', 'Equal', 'BMA-BIC', 'Softmax-LOO',
                  'Softmax-Boot', 'Stacking']:
        s = strategies.get(name, {})
        r2 = s.get('_r2', evaluate_strategy(s, base_results) if s else -np.inf)
        marker = " <<< BEST" if name == best_name else ""
        print(f"    {name:15s}: R^2_LOO={r2:+.4f}{marker}")
        s['_r2'] = r2

    return {
        'best_strategy': best_name,
        'best_config': best_config,
        'all_strategies': strategies,
    }


# ============================================================================
# Main Pipeline
# ============================================================================

def run_ensemble_for_target(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    boot_results: dict = None
) -> dict:
    """
    Full ensemble pipeline for one target variable.

    Returns:
        dict with base_results, ensemble_result, best predictions
    """
    print(f"\n{'=' * 60}")
    print(f"Target: {target_name}  (n={len(X)})")
    print(f"{'=' * 60}")

    # Train all base models
    base_results = train_all_base_models(X, y)

    # Adaptive ensemble
    ensemble_result = adaptive_ensemble(base_results, boot_results)

    # Summary
    best = ensemble_result['best_config']
    print(f"\n  >>> Best strategy: {ensemble_result['best_strategy']}")
    print(f"  >>> Weights: {best.get('weights', {})}")
    print(f"  >>> Ensemble R^2_LOO: {best.get('_r2', 'N/A')}")

    # Best single model R²
    best_single = max(base_results.items(),
                      key=lambda kv: kv[1]['r2_loo'])
    print(f"  >>> Best single model: {best_single[0]} (R^2={best_single[1]['r2_loo']})")

    return {
        'target': target_name,
        'base_results': base_results,
        'ensemble_result': ensemble_result,
        'best_single_r2': best_single[1]['r2_loo'],
        'best_ensemble_r2': best.get('_r2', -np.inf),
    }


def run_ensemble_all_targets(
    features: pd.DataFrame,
    deltas_df: pd.DataFrame,
    feature_names_list: dict = None
) -> dict:
    """
    Run ensemble for all 5 target variables (dA, dB, dxi, dH, dz_min).
    """
    targets = {
        'dA': 'dA',
        'dB': 'dB',
        'dxi': 'dxi',
        'dH': 'dH',
        'dz_min': 'dz_min'
    }

    results = {}
    summary_rows = []

    for short_name, col in targets.items():
        if col not in deltas_df.columns:
            print(f"  SKIP {short_name}: not in deltas")
            continue

        y = deltas_df[col].values

        if feature_names_list and short_name in feature_names_list:
            selected = feature_names_list[short_name]
            cols_avail = [c for c in selected if c in features.columns]
            if len(cols_avail) >= 2:
                X = np.column_stack([
                    features[c].values for c in cols_avail
                ] + [features[cols_avail[0]].values * features[cols_avail[1]].values])
            else:
                X = features[[c for c in ['lnV', 'lnM', 'Qpeak']
                              if c in features.columns]].values
        else:
            # Default features
            default_cols = [c for c in ['lnV', 'lnM', 'Qpeak']
                            if c in features.columns]
            X = features[default_cols].values

        result = run_ensemble_for_target(X, y, short_name)
        results[short_name] = result

        summary_rows.append({
            '指标': short_name,
            '最佳单模型': max(result['base_results'].items(),
                          key=lambda kv: kv[1]['r2_loo'])[0],
            '单模型R2_LOO': result['best_single_r2'],
            '最佳集成策略': result['ensemble_result']['best_strategy'],
            '集成R2_LOO': result['best_ensemble_r2'],
        })

    # Print summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    return results, summary_df


if __name__ == '__main__':
    print("BMA Ensemble module loaded. Use run_all.py to execute full pipeline.")
