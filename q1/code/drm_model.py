"""
Module: Delayed Response Model (DRM) — Physics-based river morphology evolution

Core equation (Rate Law):
  dy/dt = -beta * (y - y_e)

Equilibrium model:
  y_e = K * (S/Sref)^a * (Q/Qref)^b

Discrete recurrence:
  y_hat[i] = y_hat[i-1] * exp(-beta*dt) + y_e[i] * (1 - exp(-beta*dt))

Optimization: L-BFGS-B + 20 random starts + Differential Evolution for global search
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


def drm_predict(
    theta: tuple,
    Q_vals: np.ndarray,
    S_vals: np.ndarray,
    dt_vals: np.ndarray,
    y0: float,
    Q_ref: float,
    S_ref: float
) -> np.ndarray:
    """
    Forward prediction of DRM for n periods.

    Args:
        theta: (K, a, b, beta)
        Q_vals, S_vals: water-sediment values for each period (length n)
        dt_vals: time intervals (years)
        y0: initial observed value
        Q_ref, S_ref: reference values for normalization

    Returns:
        y_hat: predicted values at t=1..n (length n)
    """
    K, a, b, beta = theta
    n = len(Q_vals)
    y_hat = np.zeros(n)

    # Initial condition
    S_rel = S_vals[0] / S_ref
    Q_rel = Q_vals[0] / Q_ref
    y_e_prev = K * (S_rel ** a) * (Q_rel ** b)
    y_hat[0] = y0 * np.exp(-beta * dt_vals[0]) + y_e_prev * (1 - np.exp(-beta * dt_vals[0]))

    for i in range(1, n):
        S_rel = S_vals[i] / S_ref
        Q_rel = Q_vals[i] / Q_ref
        y_e = K * (S_rel ** a) * (Q_rel ** b)
        y_hat[i] = y_hat[i - 1] * np.exp(-beta * dt_vals[i]) + y_e * (1 - np.exp(-beta * dt_vals[i]))

    return y_hat


def drm_loss(
    theta: tuple,
    Q_vals: np.ndarray,
    S_vals: np.ndarray,
    dt_vals: np.ndarray,
    y_obs: np.ndarray
) -> float:
    """
    MSE loss for DRM fitting.
    y_obs[0] is initial value, y_obs[1:] are targets for periods 0..n-1
    """
    K, a, b, beta = theta
    y0 = y_obs[0]
    Q_ref = np.mean(Q_vals)
    S_ref = np.mean(S_vals)

    y_hat = drm_predict(theta, Q_vals, S_vals, dt_vals, y0, Q_ref, S_ref)
    mse = np.mean((y_obs[1:] - y_hat) ** 2)

    # Penalize extreme parameters
    penalty = 0
    if abs(a) > 2.5: penalty += 100 * (abs(a) - 2.5) ** 2
    if abs(b) > 2.5: penalty += 100 * (abs(b) - 2.5) ** 2
    if beta <= 0: penalty += 1000
    if beta > 3.0: penalty += 100 * (beta - 3.0) ** 2

    return mse + penalty


def fit_drm_lbfgsb(
    Q_vals: np.ndarray,
    S_vals: np.ndarray,
    dt_vals: np.ndarray,
    y_obs: np.ndarray,
    n_restarts: int = 20
) -> dict:
    """
    Fit DRM via L-BFGS-B with multiple random initializations.

    Returns:
        dict with best theta, loss, predictions, diagnostics
    """
    y_vals = np.abs(y_obs)  # ensure positive
    y_range = y_vals.max() - y_vals.min()

    bounds = [
        (max(0.1, y_vals.min() * 0.1), y_vals.max() * 5.0),  # K
        (-2.5, 2.5),   # a
        (-2.5, 2.5),   # b
        (0.001, 3.0)   # beta
    ]

    best_loss = np.inf
    best_theta = None

    for restart in range(n_restarts):
        # Random initial guess
        x0 = [
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(0.01, 2.0)
        ]

        try:
            res = minimize(
                drm_loss, x0,
                args=(Q_vals, S_vals, dt_vals, y_vals),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12}
            )

            if res.fun < best_loss:
                best_loss = res.fun
                best_theta = res.x
        except Exception:
            continue

    if best_theta is None:
        # Fallback
        best_theta = [np.mean(y_vals), 0.0, 0.0, 0.1]

    return _build_drm_result(best_theta, Q_vals, S_vals, dt_vals, y_vals)


def fit_drm_de(
    Q_vals: np.ndarray,
    S_vals: np.ndarray,
    dt_vals: np.ndarray,
    y_obs: np.ndarray
) -> dict:
    """
    Fit DRM via Differential Evolution for global search.
    Used as refinement after or instead of L-BFGS-B.
    """
    y_vals = np.abs(y_obs)
    bounds = [
        (max(0.1, y_vals.min() * 0.1), y_vals.max() * 5.0),
        (-2.5, 2.5), (-2.5, 2.5), (0.001, 3.0)
    ]

    try:
        res = differential_evolution(
            drm_loss,
            bounds,
            args=(Q_vals, S_vals, dt_vals, y_vals),
            maxiter=1000,
            tol=1e-8,
            seed=42,
            polish=True  # refine with L-BFGS-B
        )
        theta = res.x
    except Exception:
        theta = [np.mean(y_vals), 0.0, 0.0, 0.1]

    return _build_drm_result(theta, Q_vals, S_vals, dt_vals, y_vals)


def _build_drm_result(
    theta: np.ndarray,
    Q_vals: np.ndarray,
    S_vals: np.ndarray,
    dt_vals: np.ndarray,
    y_obs: np.ndarray
) -> dict:
    """Build result dictionary from fitted theta."""
    K, a, b, beta = theta
    Q_ref = np.mean(Q_vals)
    S_ref = np.mean(S_vals)
    y0 = y_obs[0]

    y_hat = drm_predict(theta, Q_vals, S_vals, dt_vals, y0, Q_ref, S_ref)

    ss_res = np.sum((y_obs[1:] - y_hat) ** 2)
    ss_tot = np.sum((y_obs[1:] - np.mean(y_obs[1:])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Compute y_e for each period
    y_e = np.array([
        K * (S_vals[i] / S_ref) ** a * (Q_vals[i] / Q_ref) ** b
        for i in range(len(Q_vals))
    ])

    # Stability
    is_stable = beta > 0
    tau = 1.0 / beta if beta > 1e-10 else np.inf  # relaxation time (in period units)
    memory = np.exp(-beta * np.mean(dt_vals))  # memory factor

    last_y_hat = y_hat[-1] if len(y_hat) > 0 else y0

    return {
        'K': K, 'a': a, 'b': b, 'beta': beta,
        'r2': r2, 'rmse': np.sqrt(ss_res / len(y_hat)) if len(y_hat) > 0 else np.inf,
        'is_stable': is_stable,
        'tau': tau,
        'memory_factor': memory,
        'y_obs': y_obs.tolist(),
        'y_hat': [y0] + y_hat.tolist(),  # prepend initial
        'y_e': y_e.tolist(),
        'Q_ref': Q_ref, 'S_ref': S_ref,
        'theta': theta.tolist()
    }


def fit_drm(
    Q_vals: np.ndarray,
    S_vals: np.ndarray,
    dt_vals: np.ndarray,
    y_obs: np.ndarray,
    method: str = 'hybrid'
) -> dict:
    """
    Fit DRM model. Hybrid mode: DE global → L-BFGS-B polish.
    """
    if method == 'lbfgsb':
        return fit_drm_lbfgsb(Q_vals, S_vals, dt_vals, y_obs)
    elif method == 'de':
        return fit_drm_de(Q_vals, S_vals, dt_vals, y_obs)
    else:  # hybrid
        result_de = fit_drm_de(Q_vals, S_vals, dt_vals, y_obs)
        theta_de = result_de['theta']

        # Polish with L-BFGS-B from DE solution
        y_vals = np.abs(y_obs)
        bounds = [
            (max(0.1, y_vals.min() * 0.1), y_vals.max() * 5.0),
            (-2.5, 2.5), (-2.5, 2.5), (0.001, 3.0)
        ]
        try:
            res = minimize(
                drm_loss, theta_de,
                args=(Q_vals, S_vals, dt_vals, y_vals),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12}
            )
            theta_final = res.x if res.fun < result_de['rmse'] ** 2 * len(y_vals) else theta_de
        except Exception:
            theta_final = theta_de

        return _build_drm_result(theta_final, Q_vals, S_vals, dt_vals, y_vals)


def fit_drm_all_targets(
    features: 'pd.DataFrame',
    deltas_df: 'pd.DataFrame',
    indicators_df: 'pd.DataFrame'
) -> dict:
    """
    Fit DRM for all 5 morphological indicators.

    Returns:
        dict: {target_name: drm_result}
    """
    import pandas as pd

    # Build Q, S, dt arrays
    Q_vals = features['Qavg'].values if 'Qavg' in features.columns else \
              features['V'].values / (features.get('T_days', np.ones(len(features))) * 86400)

    S_vals = features.get('Qs', features['M'].values / (features['V'].values + 1e-10))

    # dt in years — approximate from date pairs
    dates = pd.to_datetime(features['起始日期'])
    dates_end = pd.to_datetime(features['结束日期'])
    dt_vals = (dates_end - dates).dt.days.values / 365.25
    dt_vals = np.clip(dt_vals, 0.1, 2.0)

    targets = {
        'A': 'A_面积(m^2)',
        'B': 'B_水面宽(m)',
        'xi': 'xi_宽深比',
        'H': 'H_形态熵(nats)',
        'z_min': 'z_min_深泓(m)'
    }

    results = {}
    print("\n" + "=" * 60)
    print("DRM Model Fitting (Hybrid: DE + L-BFGS-B)")
    print("=" * 60)

    for short_name, col in targets.items():
        if col not in indicators_df.columns:
            print(f"  [{short_name}] SKIP: column '{col}' not found")
            continue

        y_obs = indicators_df[col].values

        if len(features) != len(y_obs) - 1:
            print(f"  [{short_name}] SKIP: feature/obs length mismatch")
            continue

        result = fit_drm(Q_vals, S_vals, dt_vals, y_obs, method='hybrid')

        results[short_name] = result

        print(f"\n  [{short_name}] DRM parameters:")
        print(f"    K={result['K']:.4f}, a={result['a']:.4f}, b={result['b']:.4f}, "
              f"beta={result['beta']:.4f}")
        print(f"    R^2={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
        print(f"    Stability: {'STABLE' if result['is_stable'] else 'UNSTABLE'}, "
              f"tau={result['tau']:.1f} periods, memory={result['memory_factor']:.4f}")
        print(f"    y_obs: {[f'{v:.2f}' for v in y_obs]}")
        print(f"    y_hat: {[f'{v:.2f}' for v in result['y_hat']]}")

    return results


def predict_future(
    drm_result: dict,
    Q_future: np.ndarray,
    S_future: np.ndarray,
    dt_future: np.ndarray,
    n_periods: int = 10
) -> np.ndarray:
    """
    Predict future trajectory assuming constant or projected Q, S.
    """
    theta = drm_result['theta']
    Q_vals = Q_future[:n_periods]
    S_vals = S_future[:n_periods]
    dt_vals = dt_future[:n_periods]
    y0 = drm_result['y_hat'][-1]  # start from last fitted value

    Q_ref = drm_result['Q_ref']
    S_ref = drm_result['S_ref']

    return drm_predict(theta, Q_vals, S_vals, dt_vals, y0, Q_ref, S_ref)
