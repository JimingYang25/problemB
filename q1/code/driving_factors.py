"""
Module 3 (v4): Feature Engineering — 14-dimension water-sediment features
Includes S-Q reconstruction + rich physical features per the BMA v4 spec.

Features extracted per interval:
  V, M, Qpeak, Qs, Cv, Vant, Mant, SDR, f_freq, rQS
Plus log-transforms and interaction terms.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_hydro_data(filepath: str) -> pd.DataFrame:
    """Load 附件1, parse datetime, handle missing values."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in ['水位(m)', '流量(m3/s)', '含沙量(kg/m3)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Loaded {len(df)} hydro records: {df['datetime'].min()} → {df['datetime'].max()}")
    return df


def reconstruct_sediment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sediment via S = a * Q^b rating curve.
    """
    valid = df.dropna(subset=['流量(m3/s)', '含沙量(kg/m3)'])
    valid = valid[(valid['流量(m3/s)'] > 0) & (valid['含沙量(kg/m3)'] > 0)]

    if len(valid) >= 5:
        log_q = np.log(valid['流量(m3/s)'].values)
        log_s = np.log(valid['含沙量(kg/m3)'].values)
        slope, intercept, r_value, _, _ = stats.linregress(log_q, log_s)
        a, b = np.exp(intercept), slope
        print(f"  S-Q rating: S = {a:.6f} * Q^{b:.4f}  (R^2={r_value**2:.4f}, n={len(valid)})")
    else:
        a, b = 1e-5, 1.5
        print(f"  WARNING: Too few valid S-Q pairs; using defaults a={a}, b={b}")

    missing = df['含沙量(kg/m3)'].isna() | (df['含沙量(kg/m3)'] <= 0)
    df.loc[missing, '含沙量(kg/m3)'] = a * (df.loc[missing, '流量(m3/s)'].clip(lower=1)) ** b
    df['含沙量(kg/m3)'] = df['含沙量(kg/m3)'].clip(lower=0.001, upper=500)

    n_filled = missing.sum()
    print(f"  Filled {n_filled}/{len(df)} ({100*n_filled/len(df):.1f}%) sediment values")
    return df, {'a': a, 'b': b}


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily (mean Q, mean S)."""
    daily = df.groupby(df['datetime'].dt.date).agg(
        Q_daily=('流量(m3/s)', 'mean'),
        S_daily=('含沙量(kg/m3)', 'mean'),
        Q_max=('流量(m3/s)', 'max'),
        Q_min=('流量(m3/s)', 'min')
    ).reset_index()
    daily.columns = ['date', 'Q', 'S', 'Q_max', 'Q_min']
    daily['date'] = pd.to_datetime(daily['date'])
    return daily


def extract_features(daily: pd.DataFrame, date_pairs: list) -> pd.DataFrame:
    """
    Extract 14 features for each interval between cross-section measurements.

    Features:
      V, M, Qpeak, Qs, Cv, Vant, Mant, SDR, f_freq, rQS
      + log transforms: lnV, lnM, lnQs, lnVant
    """
    Q95 = daily['Q'].quantile(0.95)
    eps = 1e-10
    results = []

    for start_date, end_date in date_pairs:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        mask = (daily['date'] >= start) & (daily['date'] < end)
        interval = daily[mask]

        if len(interval) < 2:
            print(f"  WARNING: Too few days in [{start_date}, {end_date})")
            continue

        T_days = len(interval)
        Q = interval['Q'].values
        S = interval['S'].values

        # ---- Basic ----
        V = np.sum(Q * 86400)
        M = np.sum(Q * S * 86400)
        Q_peak = interval['Q_max'].max()
        Qs_bar = np.mean(Q * S)

        # ---- Variability ----
        Cv = np.std(Q) / (np.mean(Q) + eps)

        # ---- Antecedent (90 days before start) ----
        ant_mask = (daily['date'] >= start - pd.Timedelta(days=90)) & (daily['date'] < start)
        ant = daily[ant_mask]
        V_ant = np.sum(ant['Q'].values * 86400) if len(ant) > 0 else V * 0.5
        M_ant = np.sum(ant['Q'].values * ant['S'].values * 86400) if len(ant) > 0 else M * 0.5

        # ---- Derived ----
        SDR = M / (V + eps)
        n_flood = np.sum(interval['Q_max'] > Q95)
        f_freq = n_flood / (T_days + 1) * 365

        if len(Q) >= 3 and len(S) >= 3:
            rQS = np.corrcoef(Q, S)[0, 1] if np.std(S) > eps else 0.0
        else:
            rQS = 0.0

        results.append({
            '起始日期': start_date, '结束日期': end_date,
            'V': V, 'M': M, 'Qpeak': Q_peak,
            'Qs': Qs_bar, 'Cv': Cv,
            'Vant': V_ant, 'Mant': M_ant,
            'SDR': SDR, 'f_freq': f_freq, 'rQS': rQS,
            'lnV': np.log(V + eps), 'lnM': np.log(M + eps),
            'lnQs': np.log(Qs_bar + eps), 'lnVant': np.log(V_ant + eps),
            'T_days': T_days
        })

    df = pd.DataFrame(results)

    # Print feature summary
    print(f"\n  Extracted {len(df)} intervals × {len(df.columns)-3} features")
    for col in ['V', 'M', 'Qpeak', 'Qs', 'Cv', 'SDR', 'f_freq', 'rQS']:
        if col in df.columns:
            print(f"    {col:8s}: {df[col].mean():.2e} ± {df[col].std():.2e}")

    return df


def select_features_for_target(
    features: pd.DataFrame,
    y: np.ndarray,
    target_name: str,
    top_k: int = 2
) -> tuple:
    """
    Select best 2 features + interaction term for a target variable.
    Uses Spearman |rho| ranking with mutual correlation filter.

    Returns:
        X_selected: (n, 3) array [f1, f2, f1*f2]
        selected_names: list of 3 feature names
    """
    feature_cols = ['lnV', 'lnM', 'lnQs', 'lnVant', 'Qpeak', 'Qs',
                    'Cv', 'SDR', 'f_freq', 'rQS', 'V', 'M', 'Vant', 'Mant']

    # Filter to columns present in features
    cols_available = [c for c in feature_cols if c in features.columns]

    # Spearman correlation with target
    spearman_rhos = {}
    for col in cols_available:
        mask = ~(np.isnan(features[col].values) | np.isnan(y))
        if mask.sum() < 3:
            continue
        rho, _ = stats.spearmanr(features[col].values[mask], y[mask])
        spearman_rhos[col] = abs(rho) if not np.isnan(rho) else 0

    # Sort by |rho|
    sorted_cols = sorted(spearman_rhos.items(), key=lambda x: x[1], reverse=True)

    # Greedy selection: pick top features with low mutual correlation
    selected = []
    for col, rho in sorted_cols:
        if len(selected) >= top_k:
            break
        # Check correlation with already selected
        ok = True
        for sel in selected:
            mask = ~(np.isnan(features[col].values) | np.isnan(features[sel].values))
            if mask.sum() < 3:
                continue
            rho_ff = abs(stats.pearsonr(features[col].values[mask],
                                        features[sel].values[mask])[0])
            if rho_ff >= 0.8:
                ok = False
                break
        if ok:
            selected.append(col)

    # Build feature matrix
    X = np.column_stack([
        features[s].values for s in selected
    ] + [features[selected[0]].values * features[selected[1]].values])

    selected_names = selected + [f'{selected[0]}*{selected[1]}']

    print(f"  [{target_name}] features: {selected_names}  "
          f"(Spearman |rho|: {[spearman_rhos.get(s, 0) for s in selected]})")

    return X, selected_names


def main():
    data_dir = Path(__file__).parent.parent / 'data_csv'
    output_dir = Path(__file__).parent.parent / 'data'

    print("=" * 60)
    print("Module 3 (v4): Feature Engineering")
    print("=" * 60)

    # Load
    hydro_file = data_dir / '附件1_逐小时水沙数据_2016-2021.csv'
    df = load_hydro_data(str(hydro_file))

    # S-Q reconstruction
    print("\n--- Sediment Reconstruction ---")
    df, rating = reconstruct_sediment(df)

    # Daily aggregation
    daily = to_daily(df)
    print(f"  Daily aggregation: {len(daily)} days")

    # Get cross-section dates
    from q1.code.preprocess import load_cross_sections
    sections_file = data_dir / '附件2_9个断面地形数据.csv'
    sections = load_cross_sections(str(sections_file))
    section_dates = sorted(sections.keys())

    date_pairs = [(section_dates[i], section_dates[i + 1])
                  for i in range(len(section_dates) - 1)]

    # Extract features
    print("\n--- Extracting Features ---")
    features = extract_features(daily, date_pairs)

    features.to_csv(output_dir / 'driving_factors.csv',
                    index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(features)} intervals to driving_factors.csv")

    return features, daily, rating


if __name__ == '__main__':
    main()
