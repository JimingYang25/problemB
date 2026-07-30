"""
Enhanced Q2 Module: Three improvements for temporal sampling performance

1. Matérn 3/2 kernel for spatial GP (better fit than exponential)
2. GP-regression interpolation for Qs reconstruction (replaces piecewise linear)
3. Active-learning sampler: pick days with highest GP predictive variance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.linalg import solve

OUTPUT_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Improvement 1: Matérn 3/2 kernel
# ============================================================================

def matern32_kernel(d: np.ndarray, sigma2: float, ell: float) -> np.ndarray:
    """Matérn ν=3/2: k(d) = sigma^2 * (1 + sqrt(3)*d/ell) * exp(-sqrt(3)*d/ell)"""
    r = np.sqrt(3) * d / ell
    return sigma2 * (1 + r) * np.exp(-r)


def fit_matern_kernel(
    distances: np.ndarray,
    empirical_cov: np.ndarray,
    n_bins: int
) -> dict:
    """Fit Matérn 3/2 kernel to empirical covariance."""
    d_flat = []
    c_flat = []
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            d_flat.append(distances[i, j])
            c_flat.append(empirical_cov[i, j])

    d_flat = np.array(d_flat)
    c_flat = np.array(c_flat)

    def loss(params):
        sigma2, ell = params
        if sigma2 <= 0 or ell <= 1e-6:
            return 1e10
        pred = matern32_kernel(d_flat, sigma2, ell)
        return np.sum((c_flat - pred) ** 2)

    sigma2_init = np.mean(np.diag(empirical_cov))
    ell_init = np.mean(d_flat) / 2

    res = minimize(loss, [sigma2_init, ell_init],
                   method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-8})
    sigma2, ell = res.x

    pred = matern32_kernel(d_flat, sigma2, ell)
    ss_res = np.sum((c_flat - pred) ** 2)
    ss_tot = np.sum((c_flat - c_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {'sigma2': sigma2, 'ell': ell, 'r2': r2, 'kernel': 'Matern32'}


# ============================================================================
# Improvement 2: GP-regression interpolation for Qs
# ============================================================================

def build_time_kernel(t: np.ndarray, sigma2: float, ell: float) -> np.ndarray:
    """
    Build GP covariance matrix for 1D time series.
    Uses Matérn 3/2 kernel on time differences.
    """
    n = len(t)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(t[i] - t[j])
            r = np.sqrt(3) * d / ell
            K[i, j] = sigma2 * (1 + r) * np.exp(-r)
    return K


def gp_interpolate_qs(
    t_full: np.ndarray,
    qs_full: np.ndarray,
    t_sampled: np.ndarray,
    qs_sampled: np.ndarray,
    noise_var: float = None
) -> np.ndarray:
    """
    GP regression to interpolate Qs from sampled days to all days.

    Args:
        t_full: day indices for the full year (0, 1, ..., 364)
        qs_full: not used for prediction, only to infer kernel params
        t_sampled: day indices where we have samples
        qs_sampled: Qs values at sampled days
        noise_var: observation noise; if None, auto-estimated

    Returns:
        qs_pred: GP posterior mean at all t_full
    """
    if len(t_sampled) < 3:
        return np.interp(t_full, t_sampled, qs_sampled)

    # Estimate kernel params from data
    qs_range = qs_full.max() - qs_full.min() + 1e-10
    sigma2 = (qs_range / 3) ** 2  # ~3 sigma covers the range
    ell = max(5, len(t_full) / 10)  # ~36 days for annual data

    if noise_var is None:
        noise_var = sigma2 * 0.01  # 1% noise

    # Build GP
    t_all = np.concatenate([t_sampled, t_full])
    K_all = build_time_kernel(t_all, sigma2, ell)

    n_s = len(t_sampled)
    K_ss = K_all[:n_s, :n_s] + noise_var * np.eye(n_s)
    K_xs = K_all[n_s:, :n_s]  # prediction points × sampled points

    # Solve
    try:
        alpha = solve(K_ss, qs_sampled, assume_a='pos')
        qs_pred = K_xs @ alpha
    except np.linalg.LinAlgError:
        qs_pred = np.interp(t_full, t_sampled, qs_sampled)

    # Ensure non-negative
    qs_pred = np.maximum(qs_pred, 0)

    return qs_pred


def estimate_annual_load_gp(
    daily_year: pd.DataFrame,
    sample_mask: np.ndarray,
) -> float:
    """
    GP-based annual load estimation (improved from piecewise linear).
    """
    t_full = np.arange(len(daily_year))
    qs_full = daily_year['Qs'].values
    t_sampled = t_full[sample_mask]
    qs_sampled = qs_full[sample_mask]

    if len(t_sampled) < 2:
        return float(np.sum(qs_full))

    qs_pred = gp_interpolate_qs(t_full, qs_full, t_sampled, qs_sampled)
    return float(np.sum(qs_pred))


# ============================================================================
# Improvement 3: Active-learning sampling
# ============================================================================

def active_learning_sampler(
    daily: pd.DataFrame,
    budget: int = 52,
    seed_size: int = 10,
    noise_var: float = None
) -> np.ndarray:
    """
    Active-learning sampling: greedily pick days that maximize
    GP predictive variance reduction.

    Algorithm:
      1. Start with `seed_size` evenly-spaced seed points
      2. Iteratively add the day with highest GP posterior variance
      3. Update GP with the new observation
      4. Repeat until budget exhausted

    This naturally focuses samples where Qs varies most rapidly —
    around flood peaks and transitions — without needing an explicit
    threshold like H0.
    """
    n = len(daily)
    mask = np.zeros(n, dtype=bool)

    # Step 1: seed with uniform spacing
    seed_indices = np.linspace(0, n - 1, seed_size, dtype=int)
    mask[seed_indices] = True

    # GP kernel params from data
    qs = daily['Qs'].values
    qs_range = qs.max() - qs.min() + 1e-10
    sigma2 = (qs_range / 3) ** 2
    ell = max(5, n / 10)
    if noise_var is None:
        noise_var = sigma2 * 0.01

    # Remaining budget
    remaining = budget - mask.sum()
    t_all = np.arange(n)

    for _ in range(remaining):
        t_sampled = t_all[mask]
        qs_sampled = qs[mask]

        # Build full kernel matrix
        K_all = build_time_kernel(t_all, sigma2, ell)
        K_noisy = K_all + noise_var * np.eye(n)

        # For each unsampled point, compute posterior variance
        # Var[f(x*)] = k(x*,x*) - k(x*,X) @ K_XX^{-1} @ k(X,x*)
        unsampled = np.where(~mask)[0]
        best_var = -1
        best_idx = -1

        # Compute inverse once
        s_idx = np.where(mask)[0]
        K_ss = K_all[np.ix_(s_idx, s_idx)] + noise_var * np.eye(len(s_idx))
        try:
            K_ss_inv = np.linalg.inv(K_ss)
        except np.linalg.LinAlgError:
            K_ss_inv = np.linalg.pinv(K_ss)

        for u in unsampled:
            k_xs = K_all[u, s_idx]
            post_var = K_all[u, u] - k_xs @ K_ss_inv @ k_xs
            if post_var > best_var:
                best_var = post_var
                best_idx = u

        mask[best_idx] = True

    print(f"  ActiveLearning: budget={budget}, seed={seed_size}, "
          f"sampled={mask.sum()}")

    return mask


# ============================================================================
# Run comparison: old vs enhanced methods
# ============================================================================

def run_enhanced_comparison():
    """Compare all sampling methods on the same data."""
    from load_data import load_hydro_timeseries, load_cross_section_velocities, normalize_and_bin
    from temporal_voi import (fit_sq_rating, build_daily_truth,
                               calibrate_h0, fixed_frequency_sampler,
                               adaptive_sampler, estimate_annual_load)
    from spatial_voi import fit_exponential_kernel

    print("=" * 70)
    print("  Enhanced Sampling: Matern32 + GP-Interp + Active-Learning")
    print("=" * 70)

    # ---- Data ----
    print("\n--- Data Loading ---")
    hydro = load_hydro_timeseries()
    rating = fit_sq_rating(hydro)
    daily = build_daily_truth(hydro, rating)
    h0 = calibrate_h0(daily)

    # ---- Spatial kernel upgrade ----
    sections = load_cross_section_velocities()
    obs_matrix, bin_centers, _ = normalize_and_bin(sections, n_bins=20)
    obs_detrended = obs_matrix - obs_matrix.mean(axis=1, keepdims=True)
    distances = np.abs(bin_centers[:, None] - bin_centers[None, :])
    empirical_cov = np.cov(obs_detrended, rowvar=False, bias=True)
    n_bins = 20

    print("\n--- Kernel Comparison ---")
    exp_kernel = fit_exponential_kernel(distances, empirical_cov, n_bins)
    mat_kernel = fit_matern_kernel(distances, empirical_cov, n_bins)

    print(f"\n  {'Kernel':12s} {'sigma^2':8s} {'ell':8s} {'R^2':8s}")
    print(f"  {'Exponential':12s} {exp_kernel['sigma2']:8.4f} "
          f"{exp_kernel['ell']:8.1f} {exp_kernel['r2']:8.4f}")
    print(f"  {'Matern 3/2':12s} {mat_kernel['sigma2']:8.4f} "
          f"{mat_kernel['ell']:8.1f} {mat_kernel['r2']:8.4f}")
    print(f"  R^2 improvement: {mat_kernel['r2'] - exp_kernel['r2']:+.4f}")

    # ---- Sampling methods comparison ----
    years = sorted(daily['date'].dt.year.unique())
    years = [y for y in years if 2016 <= y <= 2021]
    budget = 52

    print(f"\n--- Sampling Comparison (budget={budget}) ---")

    methods = []

    for year in years:
        ydf = daily[daily['date'].dt.year == year].copy()
        n_days = len(ydf)
        true_load = ydf['Qs'].sum() / 1e4

        # Method 1: Fixed + Linear interp
        fix_mask = fixed_frequency_sampler(ydf, interval_days=7, budget=budget)
        fix_lin = estimate_annual_load(ydf, fix_mask) / 1e4

        # Method 2: Adaptive + Linear interp
        adp_mask = adaptive_sampler(ydf, h0, budget_per_year=budget)
        adp_lin = estimate_annual_load(ydf, adp_mask) / 1e4

        # Method 3: Fixed + GP interp
        fix_gp = estimate_annual_load_gp(ydf, fix_mask) / 1e4

        # Method 4: Adaptive + GP interp
        adp_gp = estimate_annual_load_gp(ydf, adp_mask) / 1e4

        # Method 5: Active-Learning + GP interp
        act_mask = active_learning_sampler(ydf, budget=budget)
        act_gp = estimate_annual_load_gp(ydf, act_mask) / 1e4

        methods.append({
            'year': year,
            'true': true_load,
            'fix_lin': fix_lin, 'adp_lin': adp_lin,
            'fix_gp': fix_gp, 'adp_gp': adp_gp,
            'act_gp': act_gp
        })

    # Print results
    print(f"\n  {'Year':4s} {'True(万t)':>10s} "
          f"{'FixLin':>8s} {'AdpLin':>8s} {'FixGP':>8s} {'AdpGP':>8s} {'ActGP':>8s}")
    print(f"  {'─'*4} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    errs = {'fix_lin': [], 'adp_lin': [], 'fix_gp': [], 'adp_gp': [], 'act_gp': []}

    for m in methods:
        true = m['true']
        for key in errs:
            err = abs(m[key] - true) / true * 100
            errs[key].append(err)

        print(f"  {m['year']:4d} {true:10.0f} "
              f"{abs(m['fix_lin']-true)/true*100:7.1f}% "
              f"{abs(m['adp_lin']-true)/true*100:7.1f}% "
              f"{abs(m['fix_gp']-true)/true*100:7.1f}% "
              f"{abs(m['adp_gp']-true)/true*100:7.1f}% "
              f"{abs(m['act_gp']-true)/true*100:7.1f}%")

    print(f"\n  {'Mean Error':20s} ", end='')
    for key in errs:
        print(f"{np.mean(errs[key]):7.2f}% ", end='')
    print()

    # Best method
    best = min(errs, key=lambda k: np.mean(errs[k]))
    print(f"\n  >>> Best method: {best} (mean error = {np.mean(errs[best]):.2f}%)")

    # Save comparison
    rows = []
    for i, m in enumerate(methods):
        rows.append({
            '年份': m['year'],
            '真值(万t)': round(m['true'], 1),
            '固定+线性': round(m['fix_lin'], 1),
            '自适应+线性': round(m['adp_lin'], 1),
            '固定+GP': round(m['fix_gp'], 1),
            '自适应+GP': round(m['adp_gp'], 1),
            '主动学习+GP': round(m['act_gp'], 1),
            '固定+线性误差%': round(errs['fix_lin'][i], 1),
            '自适应+线性误差%': round(errs['adp_lin'][i], 1),
            '固定+GP误差%': round(errs['fix_gp'][i], 1),
            '自适应+GP误差%': round(errs['adp_gp'][i], 1),
            '主动学习+GP误差%': round(errs['act_gp'][i], 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        OUTPUT_DIR / 'enhanced_sampling_comparison.csv',
        index=False, encoding='utf-8-sig'
    )

    # Kernel comparison
    kernel_df = pd.DataFrame([
        {'kernel': 'Exponential', 'sigma2': exp_kernel['sigma2'],
         'ell': exp_kernel['ell'], 'R2': exp_kernel['r2']},
        {'kernel': 'Matern32', 'sigma2': mat_kernel['sigma2'],
         'ell': mat_kernel['ell'], 'R2': mat_kernel['r2']},
    ])
    kernel_df.to_csv(
        OUTPUT_DIR / 'kernel_comparison.csv',
        index=False, encoding='utf-8-sig'
    )

    print(f"\nSaved: enhanced_sampling_comparison.csv, kernel_comparison.csv")
    return methods, errs


if __name__ == '__main__':
    run_enhanced_comparison()
