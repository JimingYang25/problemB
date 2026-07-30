"""
Spatial VoI: GP spatial field model for monitoring point information value.

Steps:
  1. Build 19x20 observation matrix from depth-averaged velocities
  2. Detrend: subtract per-period mean to get stationary field
  3. Fit exponential kernel k(d) = sigma^2 * exp(-d/ell)
  4. Compute VoI_i = conditional variance ratio for each point
  5. Greedy shutdown: iteratively remove lowest-VoI point

Key formulas:
  VoI_i = (k_ii - k_{i,-i} @ K_{-i,-i}^{-1} @ k_{-i,i}) / k_ii
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.linalg import solve

DATA_DIR = Path(__file__).parent.parent.parent / 'q1' / 'data_csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_exponential_kernel(
    distances: np.ndarray,
    empirical_cov: np.ndarray,
    n_bins: int
) -> dict:
    """
    Fit k(d) = sigma^2 * exp(-d / ell) to empirical covariance via NLLS.

    Distances are in meters. Uses the off-diagonal empirical covariances.
    """
    # Extract off-diagonal pairs
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
        pred = sigma2 * np.exp(-d_flat / ell)
        return np.sum((c_flat - pred) ** 2)

    # Initial guess
    sigma2_init = np.mean(np.diag(empirical_cov))
    ell_init = np.mean(d_flat) / 2

    res = minimize(loss, [sigma2_init, ell_init],
                   method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-8})
    sigma2, ell = res.x

    # R^2 of kernel fit
    pred = sigma2 * np.exp(-d_flat / ell)
    ss_res = np.sum((c_flat - pred) ** 2)
    ss_tot = np.sum((c_flat - c_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  GP Kernel fit: k(d) = {sigma2:.4f} * exp(-d / {ell:.1f})")
    print(f"    sigma^2 = {sigma2:.4f}")
    print(f"    ell     = {ell:.1f} m")
    print(f"    R^2     = {r2:.4f}")

    return {'sigma2': sigma2, 'ell': ell, 'r2': r2}


def compute_voi(
    K: np.ndarray,
    noise_var: float = 0.0
) -> np.ndarray:
    """
    Compute VoI for each of n points in the network.

    VoI_i = (K_ii - K_{i,-i} @ inv(K_{-i,-i} + noise*I) @ K_{-i,i}) / K_ii

    Returns VoI array of length n, each in [0, 1].
    """
    n = K.shape[0]
    K_noisy = K.copy()
    np.fill_diagonal(K_noisy, K_noisy.diagonal() + noise_var)

    voi = np.zeros(n)

    for i in range(n):
        # Build K_{-i,-i} by removing row i and column i
        idx = [j for j in range(n) if j != i]
        K_sub = K_noisy[np.ix_(idx, idx)]
        k_cross = K_noisy[i, idx]  # K_{i, -i}

        # solve instead of inv for stability
        try:
            alpha = solve(K_sub, k_cross, assume_a='pos')
            cond_var = K_noisy[i, i] - k_cross @ alpha
            voi[i] = cond_var / K_noisy[i, i]
        except np.linalg.LinAlgError:
            voi[i] = 1.0  # if singular, point is irreplaceable

    voi = np.clip(voi, 0.0, 1.0)

    return voi


def greedy_shutdown(
    K: np.ndarray,
    bin_centers: np.ndarray,
    n_shutdown: int,
    noise_var: float = 0.0
) -> list:
    """
    Greedy shutdown: at each step, remove the point with minimum VoI,
    recompute VoI for remaining points.

    Returns list of (step, position_m, normalized_position, VoI_at_removal).
    """
    n = K.shape[0]
    active = list(range(n))
    shutdown_log = []

    for step in range(n_shutdown):
        # Compute VoI for all active points
        K_active = K[np.ix_(active, active)]
        voi = compute_voi(K_active, noise_var=noise_var)

        # Find the one with minimum VoI
        worst_idx = np.argmin(voi)
        worst_global = active[worst_idx]

        shutdown_log.append({
            'step': step + 1,
            'position_m': round(bin_centers[worst_global], 0),
            'position_normalized': round(worst_global / (n - 1), 3),
            'voi': round(voi[worst_idx], 4)
        })

        active.pop(worst_idx)

        print(f"  Step {step+1}: Shutdown x={bin_centers[worst_global]:.0f}m "
              f"(bin {worst_global}), VoI={voi[worst_idx]:.4f}")

    return shutdown_log, active


def run_spatial_voi(n_bins: int = 20, n_shutdown: int = 6) -> dict:
    """
    Full spatial VoI pipeline.
    """
    from load_data import load_cross_section_velocities, normalize_and_bin

    print("=" * 60)
    print("Q2 — Spatial VoI: GP Model + Monitoring Point Shutdown")
    print("=" * 60)

    # Load and bin
    print("\n--- Loading & Binning ---")
    sections = load_cross_section_velocities()
    obs_matrix, bin_centers, dates = normalize_and_bin(sections, n_bins=n_bins)

    # Detrend: subtract per-period mean
    obs_detrended = obs_matrix - obs_matrix.mean(axis=1, keepdims=True)

    # Empirical covariance matrix (19 periods × 20 bins → 20×20 cov)
    empirical_cov = np.cov(obs_detrended, rowvar=False, bias=True)
    section_width = bin_centers[-1] - bin_centers[0]
    avg_width = section_width

    # Distance matrix between bin centers (in meters)
    distances = np.abs(bin_centers[:, None] - bin_centers[None, :])

    # Fit exponential kernel
    print("\n--- Fitting GP Kernel ---")
    kernel = fit_exponential_kernel(distances, empirical_cov, n_bins)

    # Build K matrix
    sigma2, ell = kernel['sigma2'], kernel['ell']
    K = sigma2 * np.exp(-distances / ell)

    # Noise variance (15% of field variance per README)
    noise_var = 0.15 * sigma2
    print(f"    noise_var = {noise_var:.4f} ({100*noise_var/sigma2:.0f}% of sigma^2)")
    print(f"    Section width ~ {avg_width:.0f} m")

    # Full-network VoI
    print("\n--- Full-Network VoI ---")
    voi_full = compute_voi(K, noise_var=noise_var)
    for i in range(n_bins):
        print(f"  Bin {i:2d} (x={bin_centers[i]:6.0f}m): VoI={voi_full[i]:.4f}")

    # Greedy shutdown (30% of 20 = 6)
    print(f"\n--- Greedy Shutdown ({n_shutdown}/{n_bins} = 30%) ---")
    shutdown_log, remaining = greedy_shutdown(
        K, bin_centers, n_shutdown, noise_var=noise_var
    )

    # Remaining points
    print(f"\n  Remaining {len(remaining)} points:")
    for idx in remaining:
        print(f"    x={bin_centers[idx]:.0f}m (bin {idx})")

    # Save
    shutdown_df = pd.DataFrame(shutdown_log)
    shutdown_df.to_csv(
        OUTPUT_DIR / 'shutdown_sequence.csv',
        index=False, encoding='utf-8-sig'
    )

    # Save kernel params
    with open(OUTPUT_DIR / 'spatial_kernel.txt', 'w', encoding='utf-8') as f:
        f.write(f"sigma^2 = {sigma2:.6f}\n")
        f.write(f"ell = {ell:.4f} m\n")
        f.write(f"R^2 = {kernel['r2']:.4f}\n")
        f.write(f"noise_var = {noise_var:.6f}\n")
        f.write(f"section_width = {avg_width:.0f} m\n")
        f.write(f"VoI definition: conditional variance ratio\n")

    print(f"\nSaved: shutdown_sequence.csv, spatial_kernel.txt")

    return {
        'kernel': kernel,
        'voi_full': voi_full,
        'shutdown_log': shutdown_log,
        'remaining': remaining,
        'K': K,
        'bin_centers': bin_centers,
        'empirical_cov': empirical_cov,
        'distances': distances,
    }


if __name__ == '__main__':
    run_spatial_voi()
