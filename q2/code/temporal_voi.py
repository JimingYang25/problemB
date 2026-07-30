"""
Temporal VoI: Adaptive Sampling Rule for Sediment Load Estimation

Steps:
  1. S-Q rating curve reconstruction for missing sediment
  2. |dH/dt| threshold H0 calibration (80th percentile)
  3. Fixed-frequency sampler (every 7 days, N=52/year)
  4. Adaptive sampler: 40% baseline + 60% event-driven
  5. Compare annual sediment load estimation errors
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).parent.parent.parent / 'q1' / 'data_csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_sq_rating(hydro_df: pd.DataFrame) -> dict:
    """
    Fit S = a * Q^b via log-linear regression on valid (Q,S) pairs.
    """
    valid = hydro_df.dropna(subset=['流量(m3/s)', '含沙量(kg/m3)'])
    valid = valid[(valid['流量(m3/s)'] > 0) & (valid['含沙量(kg/m3)'] > 0)]

    log_q = np.log(valid['流量(m3/s)'].values)
    log_s = np.log(valid['含沙量(kg/m3)'].values)

    slope, intercept, r_value, _, _ = stats.linregress(log_q, log_s)
    a = np.exp(intercept)
    b = slope
    r2 = r_value ** 2

    print(f"  S-Q rating: S = {a:.6f} * Q^{b:.4f}")
    print(f"  R^2 = {r2:.4f}, n = {len(valid)}")

    return {'a': a, 'b': b, 'r2': r2, 'n': len(valid)}


def build_daily_truth(hydro_df: pd.DataFrame, rating: dict) -> pd.DataFrame:
    """
    Build daily truth: use measured S where available, fill gaps with S-Q rating.
    Compute daily sediment flux Qs (tons/day) = Q * S * 86.4.
    """
    df = hydro_df.copy()
    a, b = rating['a'], rating['b']

    # Fill missing S
    missing = df['含沙量(kg/m3)'].isna() | (df['含沙量(kg/m3)'] <= 0)
    df['S_filled'] = df['含沙量(kg/m3)'].copy()
    df.loc[missing, 'S_filled'] = a * (df.loc[missing, '流量(m3/s)'].clip(lower=1)) ** b
    df['S_filled'] = df['S_filled'].clip(lower=0.001, upper=500)

    # Daily aggregation
    daily = df.set_index('datetime').resample('D').agg({
        '水位(m)': 'mean',
        '流量(m3/s)': 'mean',
        'S_filled': 'mean',
    }).reset_index()

    daily.columns = ['date', 'H', 'Q', 'S']
    daily['dH'] = daily['H'].diff().abs()
    daily['Qs'] = daily['Q'] * daily['S'] * 86.4  # tons/day

    # Fill leading NaN in dH
    daily['dH'] = daily['dH'].fillna(0)

    print(f"  Daily truth: {len(daily)} days, "
          f"Qs=[{daily['Qs'].min():.1f}, {daily['Qs'].max():.1f}] t/day")

    return daily


def calibrate_h0(daily: pd.DataFrame, quantile: float = 0.80) -> float:
    """
    Calibrate H0 threshold as the specified quantile of |dH/dt|.
    """
    h0 = daily['dH'].quantile(quantile)
    print(f"  H0 = {h0:.4f} m/day ({100*quantile:.0f}th percentile of |dH/dt|)")
    return h0


def fixed_frequency_sampler(
    daily: pd.DataFrame,
    interval_days: int = 7,
    budget: int = 52
) -> np.ndarray:
    """
    Fixed-frequency sampling: every `interval_days` days.
    Adjusts to match exact budget.
    """
    n = len(daily)
    mask = np.zeros(n, dtype=bool)

    # Try to hit exact budget
    if budget > 0 and n > budget:
        step = max(1, n // budget)
        indices = np.arange(0, n, step)[:budget]
        mask[indices] = True
    else:
        mask[::interval_days] = True

    return mask


def adaptive_sampler(
    daily: pd.DataFrame,
    h0: float,
    budget_per_year: int = 52,
    baseline_frac: float = 0.40
) -> np.ndarray:
    """
    Adaptive sampling based on |dH/dt| exceeding H0.

    Strategy:
      - baseline_frac of budget: uniformly spread across year
      - remaining budget: allocated to events (|dH/dt| > H0),
        one sample per event peak sorted by intensity,
        repeated round-robin if budget allows.

    Returns boolean mask of sampled days (length = len(daily)).
    """
    n_total = len(daily)
    mask = np.zeros(n_total, dtype=bool)

    # ---- Baseline: uniform spacing ----
    n_baseline = max(1, int(np.ceil(budget_per_year * baseline_frac)))
    baseline_step = max(1, n_total // n_baseline)
    mask[::baseline_step] = True

    # ---- Event detection ----
    above = daily['dH'].values > h0
    events = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i
            if end > start:
                peak_val = daily['dH'].values[start:end].max()
                events.append({
                    'start': start, 'end': end,
                    'peak': peak_val, 'n_days': end - start
                })
        else:
            i += 1

    if events:
        # Sort by peak intensity (descending)
        events.sort(key=lambda e: e['peak'], reverse=True)

        # ---- Event allocation: one sample per event, round-robin ----
        n_event_budget = max(0, budget_per_year - mask.sum())
        allocated = 0
        round_num = 0
        max_rounds = 10  # safety limit

        while allocated < n_event_budget and round_num < max_rounds:
            made_progress = False
            for ev in events:
                if allocated >= n_event_budget:
                    break
                # In round 0: sample the peak day
                # In round 1: sample the 2nd-highest dH day, etc.
                ev_days = list(range(ev['start'], ev['end']))
                ev_days.sort(key=lambda d: -daily['dH'].values[d])
                if round_num < len(ev_days):
                    d = ev_days[round_num]
                    if not mask[d]:
                        mask[d] = True
                        allocated += 1
                        made_progress = True
            if not made_progress:
                break
            round_num += 1

    n_sampled = mask.sum()
    if n_sampled != budget_per_year:
        # Adjust to exact budget by adding/removing from baseline
        if n_sampled < budget_per_year:
            extra = budget_per_year - n_sampled
            # Add midpoints between existing samples
            sampled_idx = np.where(mask)[0]
            for _ in range(extra):
                if len(sampled_idx) < 2:
                    break
                gaps = np.diff(sampled_idx)
                widest = np.argmax(gaps)
                mid = sampled_idx[widest] + gaps[widest] // 2
                if not mask[mid]:
                    mask[mid] = True
                    sampled_idx = np.where(mask)[0]
        else:
            # Remove excess from baseline (least valuable days)
            excess = n_sampled - budget_per_year
            baseline_idx = np.where(mask)[0]
            to_remove = baseline_idx[:: max(1, len(baseline_idx) // excess)][:excess]
            mask[to_remove] = False

    print(f"  Adaptive: budget={budget_per_year}, sampled={mask.sum()}, "
          f"baseline={n_baseline}, n_events={len(events)}")

    return mask


def estimate_annual_load(
    daily_year: pd.DataFrame,
    sample_mask: np.ndarray,
) -> float:
    """
    Estimate annual sediment load from sampled days using piecewise
    linear interpolation of Qs, then integrate.

    Args:
        daily_year: one year's daily data
        sample_mask: boolean mask for THAT year's data
    """
    if len(daily_year) == 0:
        return 0.0

    day_indices = np.arange(len(daily_year))
    sampled_indices = day_indices[sample_mask]
    sampled_qs = daily_year['Qs'].values[sample_mask]

    if len(sampled_indices) < 2:
        return float(np.sum(daily_year['Qs'].values))  # fallback

    qs_interp = np.interp(day_indices, sampled_indices, sampled_qs)
    return float(np.sum(qs_interp))


def validate_sampling(
    daily: pd.DataFrame,
    h0: float,
    budget: int = 52
) -> dict:
    """
    Compare fixed-frequency vs adaptive sampling for each year 2016-2021.
    """
    years = sorted(daily['date'].dt.year.unique())
    years = [y for y in years if 2016 <= y <= 2021]

    results = []
    print(f"\n  {'Year':6s} {'True(万t)':>10s} {'Fixed(万t)':>10s} "
          f"{'FixErr%':>8s} {'Adapt(万t)':>10s} {'AdpErr%':>8s} "
          f"{'Winner':>8s}")

    fix_errors = []
    adp_errors = []

    for year in years:
        ydf = daily[daily['date'].dt.year == year]

        # True annual load
        true_load = ydf['Qs'].sum() / 1e4  # 万吨

        # Generate masks for this year only
        fix_mask = fixed_frequency_sampler(ydf, interval_days=7, budget=budget)
        adp_mask = adaptive_sampler(ydf, h0, budget_per_year=budget)

        # Estimate (pass year-specific data and mask)
        fix_load = estimate_annual_load(ydf, fix_mask) / 1e4
        adp_load = estimate_annual_load(ydf, adp_mask) / 1e4

        # Errors
        fix_err = abs(fix_load - true_load) / true_load * 100 if true_load > 0 else 0
        adp_err = abs(adp_load - true_load) / true_load * 100 if true_load > 0 else 0

        fix_errors.append(fix_err)
        adp_errors.append(adp_err)

        winner = "Adaptive" if adp_err < fix_err else "Fixed"

        results.append({
            '年份': year,
            '真值(万吨)': round(true_load, 1),
            '固定频率估计': round(fix_load, 1),
            '固定频率误差(%)': round(fix_err, 1),
            '自适应估计': round(adp_load, 1),
            '自适应误差(%)': round(adp_err, 1),
            '优胜': winner
        })

        print(f"  {year:4d}  {true_load:10.1f}  {fix_load:10.1f}  "
              f"{fix_err:7.1f}%  {adp_load:10.1f}  {adp_err:7.1f}%  "
              f"{winner:>8s}")

    mean_fix = np.mean(fix_errors)
    mean_adp = np.mean(adp_errors)

    print(f"\n  Mean absolute error: Fixed={mean_fix:.2f}%, Adaptive={mean_adp:.2f}%")
    print(f"  Relative improvement: {(1-mean_adp/mean_fix)*100:.0f}%")

    return {
        'results': pd.DataFrame(results),
        'mean_fix_error': round(mean_fix, 2),
        'mean_adp_error': round(mean_adp, 2),
        'improvement_pct': round((1 - mean_adp / mean_fix) * 100, 0) if mean_fix > 0 else 0
    }


def run_temporal_voi() -> dict:
    """Full temporal VoI pipeline."""
    from load_data import load_hydro_timeseries

    print("=" * 60)
    print("Q2 — Temporal VoI: Adaptive Sampling Validation")
    print("=" * 60)

    # Load
    print("\n--- Loading Hydro Data ---")
    hydro = load_hydro_timeseries()

    # S-Q rating
    print("\n--- S-Q Rating Curve ---")
    rating = fit_sq_rating(hydro)

    with open(OUTPUT_DIR / 'rating_curve.txt', 'w', encoding='utf-8') as f:
        f.write(f"S = a * Q^b\n")
        f.write(f"a = {rating['a']:.6f}\n")
        f.write(f"b = {rating['b']:.4f}\n")
        f.write(f"R^2 = {rating['r2']:.4f}\n")
        f.write(f"n_valid = {rating['n']}\n")

    # Build daily truth
    print("\n--- Building Daily Truth ---")
    daily = build_daily_truth(hydro, rating)

    # Save
    daily.to_csv(
        Path(__file__).parent.parent / 'data' / 'daily_series.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"  Saved daily_series.csv")

    # H0 calibration
    print("\n--- H0 Threshold Calibration ---")
    h0 = calibrate_h0(daily, quantile=0.80)

    # Validate
    print(f"\n--- Sampling Validation (budget={52} samples/year) ---")
    validation = validate_sampling(daily, h0, budget=52)

    # Save
    validation['results'].to_csv(
        OUTPUT_DIR / 'sampling_validation.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"\nSaved: sampling_validation.csv, rating_curve.txt")

    return {
        'rating': rating,
        'h0': h0,
        'validation': validation,
        'daily': daily,
    }


if __name__ == '__main__':
    run_temporal_voi()
