"""
Optimization Module (论文方案优化):
  1. Multi-objective GA for monitoring point selection (min error + min cost)
  2. KL-divergence based adaptive trigger
  3. Compare GA vs greedy VoI for shutdown decisions

Paper reference: NSGA-style Pareto optimization + information-theoretic trigger
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon

OUTPUT_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Part 1: Multi-Objective GA for Sensor Placement
# ============================================================================

def reconstruct_field_from_selected(
    K: np.ndarray,
    selected: list,
    full_obs: np.ndarray,
    noise_var: float = 0.0
) -> np.ndarray:
    """
    GP-based reconstruction of full field from selected observation points.
    Uses conditional expectation of GP given observations at 'selected' indices.

    Returns reconstructed field values at all n positions.
    """
    n = K.shape[0]
    K_noisy = K.copy()
    np.fill_diagonal(K_noisy, K_noisy.diagonal() + noise_var)

    # Mean of all periods' observations at selected points
    obs_mean = np.mean(full_obs[:, selected], axis=0)

    # GP prediction at all points given selected observations
    K_ss = K_noisy[np.ix_(selected, selected)]
    K_xs = K_noisy[:, selected]  # n × n_selected

    try:
        alpha = np.linalg.solve(K_ss, obs_mean)
        reconstruction = K_xs @ alpha
    except np.linalg.LinAlgError:
        reconstruction = np.zeros(n)
        for i, s in enumerate(selected):
            reconstruction[s] = obs_mean[i]

    return reconstruction


def compute_reconstruction_error(
    K: np.ndarray,
    selected: list,
    full_obs: np.ndarray,
    noise_var: float = 0.0
) -> float:
    """
    Compute relative reconstruction error using GP conditional mean.
    Error = ||true - reconstructed|| / ||true|| (averaged over periods)
    """
    n_periods, n = full_obs.shape
    errors = []

    for t in range(n_periods):
        obs_mean = full_obs[t, selected]
        K_ss = K[np.ix_(selected, selected)].copy()
        np.fill_diagonal(K_ss, K_ss.diagonal() + noise_var)
        K_xs = K[:, selected]

        try:
            alpha = np.linalg.solve(K_ss, obs_mean)
            recon = K_xs @ alpha
        except np.linalg.LinAlgError:
            recon = np.zeros(n)
            for i, s in enumerate(selected):
                recon[s] = obs_mean[i]

        err = np.linalg.norm(full_obs[t] - recon) / (np.linalg.norm(full_obs[t]) + 1e-10)
        errors.append(err)

    return float(np.mean(errors))


def ga_optimize_sensors(
    K: np.ndarray,
    obs_matrix: np.ndarray,
    n_total: int,
    n_min: int = 2,
    n_max: int = None,
    pop_size: int = 50,
    n_generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    noise_var: float = 0.0,
    seed: int = 42
) -> dict:
    """
    Multi-objective GA for sensor placement.

    Objectives:
      1. Minimize reconstruction error
      2. Minimize number of sensors (cost)

    Uses weighted-sum scalarization: F(x) = w_err * E(x) + w_cost * C(x)
    with weight sweep to trace Pareto front.

    Each individual is a binary mask of length n_total.
    """
    rng = np.random.RandomState(seed)
    if n_max is None:
        n_max = n_total

    # ---- Weight sweep for Pareto front ----
    weight_pairs = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]

    pareto_results = []

    for w_err, w_cost in weight_pairs:
        # Initialize population
        population = np.zeros((pop_size, n_total), dtype=bool)
        for i in range(pop_size):
            n_sensors = rng.randint(n_min, n_max + 1)
            idx = rng.choice(n_total, size=n_sensors, replace=False)
            population[i, idx] = True
            # Always include edges (boundary constraint from paper)
            population[i, 0] = True
            population[i, -1] = True

        for gen in range(n_generations):
            # Evaluate fitness
            fitness = np.zeros(pop_size)
            for i in range(pop_size):
                selected = list(np.where(population[i])[0])
                if len(selected) < n_min:
                    fitness[i] = np.inf
                    continue
                err = compute_reconstruction_error(
                    K, selected, obs_matrix, noise_var
                )
                cost_ratio = len(selected) / n_total
                fitness[i] = w_err * err + w_cost * cost_ratio

            # Tournament selection
            parents = np.zeros_like(population)
            for i in range(pop_size):
                t1, t2 = rng.choice(pop_size, 2, replace=False)
                parents[i] = population[t1] if fitness[t1] < fitness[t2] else population[t2]

            # Crossover
            offspring = parents.copy()
            for i in range(0, pop_size - 1, 2):
                if rng.rand() < crossover_rate:
                    point = rng.randint(1, n_total - 1)
                    offspring[i, point:] = parents[i + 1, point:]
                    offspring[i + 1, point:] = parents[i, point:]

            # Mutation
            for i in range(pop_size):
                for j in range(1, n_total - 1):  # preserve edges
                    if rng.rand() < mutation_rate:
                        offspring[i, j] = not offspring[i, j]

            # Ensure constraints
            for i in range(pop_size):
                offspring[i, 0] = True
                offspring[i, -1] = True
                if offspring[i].sum() < n_min:
                    # Add random sensors to meet minimum
                    need = n_min - int(offspring[i].sum())
                    available = np.where(~offspring[i])[0]
                    if len(available) >= need:
                        offspring[i, rng.choice(available, need, replace=False)] = True

            population = offspring

        # Best solution for this weight pair
        best_idx = np.argmin(fitness)
        best_selected = list(np.where(population[best_idx])[0])
        best_err = compute_reconstruction_error(
            K, best_selected, obs_matrix, noise_var
        )
        best_cost = len(best_selected)

        pareto_results.append({
            'w_err': w_err,
            'w_cost': w_cost,
            'n_sensors': best_cost,
            'error': round(best_err, 6),
            'selected': best_selected,
            'fitness': fitness[best_idx]
        })

        print(f"  GA (w_err={w_err}, w_cost={w_cost}): "
              f"n={best_cost}, error={best_err:.4f}, "
              f"selected={best_selected}")

    return {
        'pareto': pareto_results,
        'pop_size': pop_size,
        'n_generations': n_generations
    }


# ============================================================================
# Part 2: KL-Divergence Adaptive Trigger
# ============================================================================

def compute_kl_trigger(
    daily: pd.DataFrame,
    window_days: int = 30,
    theta: float = 0.15
) -> dict:
    """
    KL-divergence based adaptive sampling trigger.

    Algorithm:
      1. Define baseline profile P_ref as the velocity distribution
         over the past `window_days` days (steady-state reference)
      2. Compute KL divergence of current-day profile vs baseline
      3. If D_KL > theta → trigger high-frequency sampling
      4. Continue until D_KL < theta/2 for 3 consecutive windows

    Uses Jensen-Shannon divergence (symmetric, bounded [0, ln2])
    as a robust proxy for KL divergence.

    Returns:
        dict with trigger_dates, kl_values, baseline_info
    """
    n = len(daily)
    kl_values = np.zeros(n)
    trigger = np.zeros(n, dtype=bool)

    # Use |dH| and Q distributions as the profile features
    # (proxy for velocity profile when full vertical data unavailable)
    for t in range(window_days, n):
        # Baseline: distribution of |dH| in past window
        baseline_window = daily['dH'].values[t - window_days:t]
        # Current: relative change in the most recent few days
        current_window = daily['dH'].values[max(0, t - 5):t + 1]

        if len(baseline_window) < 5 or len(current_window) < 3:
            continue

        # Build histograms
        bins = np.linspace(0, max(baseline_window.max(), current_window.max()) + 1e-10, 20)
        hist_baseline, _ = np.histogram(baseline_window, bins=bins, density=True)
        hist_current, _ = np.histogram(current_window, bins=bins, density=True)

        # Add small constant for numerical stability
        hist_baseline = hist_baseline + 1e-10
        hist_current = hist_current + 1e-10

        # Jensen-Shannon divergence
        js_div = jensenshannon(hist_baseline, hist_current)
        kl_values[t] = js_div

        if js_div > theta:
            trigger[t] = True

    # Mark exit from trigger state (continuous 3 below theta/2)
    in_trigger = False
    below_count = 0
    trigger_final = np.zeros(n, dtype=bool)
    for t in range(n):
        if kl_values[t] > theta:
            in_trigger = True
            below_count = 0
        elif in_trigger and kl_values[t] < theta / 2:
            below_count += 1
            if below_count >= 3:
                in_trigger = False
                below_count = 0

        trigger_final[t] = in_trigger

    n_triggered = trigger_final.sum()
    trigger_frac = 100 * n_triggered / n

    print(f"  KL Trigger: theta={theta}, window={window_days}d")
    print(f"    Triggered days: {n_triggered}/{n} ({trigger_frac:.1f}%)")
    print(f"    H0-based (>0.084 m/day): "
          f"{(daily['dH'].values > 0.084).sum()}/{n}  "
          f"({100*(daily['dH'].values > 0.084).sum()/n:.1f}%)")

    return {
        'trigger_dates': trigger_final,
        'kl_values': kl_values.tolist(),
        'n_triggered': int(n_triggered),
        'theta': theta,
        'window_days': window_days
    }


def kl_adaptive_sampler(
    daily: pd.DataFrame,
    kl_result: dict,
    budget_per_year: int = 52,
    baseline_frac: float = 0.40
) -> np.ndarray:
    """
    Adaptive sampler that uses KL-divergence trigger instead of |dH/dt| > H0.

    Baseline + KL-triggered event-driven sampling.
    """
    n_total = len(daily)
    mask = np.zeros(n_total, dtype=bool)

    # Baseline
    n_baseline = max(1, int(np.ceil(budget_per_year * baseline_frac)))
    baseline_step = max(1, n_total // n_baseline)
    mask[::baseline_step] = True

    # KL-triggered events
    trigger = kl_result['trigger_dates']

    # Find continuous triggered intervals
    events = []
    i = 0
    while i < n_total:
        if trigger[i]:
            start = i
            while i < n_total and trigger[i]:
                i += 1
            end = i
            if end > start:
                events.append({'start': start, 'end': end,
                               'n_days': end - start})
        else:
            i += 1

    events.sort(key=lambda e: e['n_days'], reverse=True)

    # Allocate to triggered events
    n_event_budget = max(0, budget_per_year - mask.sum())
    allocated = 0
    for ev in events:
        if allocated >= n_event_budget:
            break
        # Sample the peak |dH| day in each event
        peak_day = ev['start'] + np.argmax(
            daily['dH'].values[ev['start']:ev['end']]
        )
        if not mask[peak_day]:
            mask[peak_day] = True
            allocated += 1

    # Fill remaining budget
    if mask.sum() < budget_per_year:
        gap = budget_per_year - mask.sum()
        unsampled = np.where(~mask)[0]
        step = max(1, len(unsampled) // gap)
        for i in range(0, len(unsampled), step):
            if mask.sum() >= budget_per_year:
                break
            mask[unsampled[i]] = True

    print(f"  KL-Adaptive: budget={budget_per_year}, sampled={mask.sum()}, "
          f"events={len(events)}")

    return mask


# ============================================================================
# Part 3: Compare Methods
# ============================================================================

def compare_shutdown_methods(
    K: np.ndarray,
    obs_matrix: np.ndarray,
    bin_centers: np.ndarray,
    n_shutdown: int = 6,
    noise_var: float = 0.0
) -> dict:
    """
    Compare greedy VoI shutdown vs GA-optimized shutdown.
    """
    from spatial_voi import compute_voi, greedy_shutdown

    n = len(bin_centers)

    print("=" * 60)
    print("Comparing Shutdown Methods: Greedy VoI vs GA Optimization")
    print("=" * 60)

    # --- Method 1: Greedy VoI ---
    print("\n--- Greedy VoI Shutdown ---")
    vo_shutdown, vo_remaining = greedy_shutdown(
        K, bin_centers, n_shutdown, noise_var=noise_var
    )
    vo_error = compute_reconstruction_error(
        K, vo_remaining, obs_matrix, noise_var
    )

    # --- Method 2: GA Optimization ---
    print(f"\n--- GA Optimization (pop={50}, gen={100}) ---")
    ga_result = ga_optimize_sensors(
        K, obs_matrix, n,
        n_min=n - n_shutdown,
        n_max=n - n_shutdown + 2,
        pop_size=50,
        n_generations=100,
        noise_var=noise_var
    )

    # Find GA solution closest to n - n_shutdown sensors
    target_n = n - n_shutdown
    ga_best = min(ga_result['pareto'],
                  key=lambda p: abs(p['n_sensors'] - target_n))

    print(f"\n{'='*60}")
    print(f"Comparison @ {target_n} sensors (shutdown {n_shutdown} = 30%):")
    print(f"  Greedy VoI: error={vo_error:.4f}, n_sensors={len(vo_remaining)}")
    print(f"  GA Opt:     error={ga_best['error']:.4f}, n_sensors={ga_best['n_sensors']}")
    print(f"  GA selected positions: {sorted(ga_best['selected'])}")

    # Save Pareto front
    pareto_df = pd.DataFrame([{
        'w_err': p['w_err'],
        'w_cost': p['w_cost'],
        'n_sensors': p['n_sensors'],
        'error': p['error']
    } for p in ga_result['pareto']])
    pareto_df.to_csv(
        OUTPUT_DIR / 'ga_pareto_front.csv',
        index=False, encoding='utf-8-sig'
    )

    return {
        'greedy_voi': {
            'error': vo_error,
            'n_sensors': len(vo_remaining),
            'shutdown': vo_shutdown
        },
        'ga': ga_result,
        'ga_best': ga_best
    }


def run_optimized_pipeline():
    """Run full optimized pipeline: GA sensor placement + KL trigger."""
    from load_data import load_cross_section_velocities, normalize_and_bin

    print("=" * 70)
    print("  Q2 Optimization: GA Sensor Placement + KL Adaptive Trigger")
    print("=" * 70)

    # Load data
    sections = load_cross_section_velocities()
    obs_matrix, bin_centers, dates = normalize_and_bin(sections, n_bins=20)
    obs_detrended = obs_matrix - obs_matrix.mean(axis=1, keepdims=True)

    # Build K from spatial_voi kernel params
    from spatial_voi import fit_exponential_kernel
    distances = np.abs(bin_centers[:, None] - bin_centers[None, :])
    empirical_cov = np.cov(obs_detrended, rowvar=False, bias=True)
    kernel = fit_exponential_kernel(distances, empirical_cov, n_bins=20)
    K = kernel['sigma2'] * np.exp(-distances / kernel['ell'])
    noise_var = 0.15 * kernel['sigma2']

    # Compare shutdown methods
    comparison = compare_shutdown_methods(
        K, obs_detrended, bin_centers,
        n_shutdown=6, noise_var=noise_var
    )

    # KL trigger analysis
    from load_data import load_hydro_timeseries
    from temporal_voi import fit_sq_rating, build_daily_truth
    hydro = load_hydro_timeseries()
    rating = fit_sq_rating(hydro)
    daily = build_daily_truth(hydro, rating)

    print(f"\n{'='*60}")
    print("KL-Divergence Adaptive Trigger")
    print(f"{'='*60}")

    # Scan theta values to find reasonable trigger rates
    for theta_test in [0.15, 0.30, 0.40, 0.50]:
        kl_test = compute_kl_trigger(daily, window_days=30, theta=theta_test)
    # Use theta=0.40 as a balanced threshold (~15-25% trigger rate)
    kl_result = compute_kl_trigger(daily, window_days=30, theta=0.40)

    # Save KL trigger analysis
    kl_df = pd.DataFrame({
        'date': daily['date'],
        'kl_divergence': kl_result['kl_values'],
        'trigger': kl_result['trigger_dates'].astype(int)
    })
    kl_df.to_csv(
        OUTPUT_DIR / 'kl_trigger_analysis.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"\nSaved: ga_pareto_front.csv, kl_trigger_analysis.csv")

    return comparison, kl_result


if __name__ == '__main__':
    run_optimized_pipeline()
