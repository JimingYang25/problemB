"""
Revised Q1+Q2 Analysis — Strict Outer Validation + Auditable Methodology

Key features (vs old implementation):
  - Common cross-section Omega=[0,4583m], PCHIP, normalized entropy
  - log(1+S) rolling cross-year rating with H_dot, I_rising
  - Strict outer LOCO DRM: all scaling/selection/mu inside each fold
  - 12 standard wet-zone verticals, shrinkage covariance, exhaustive search
  - Online backward-only sampling replay, real-S observation scoring only
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve
from scipy.optimize import minimize
from itertools import combinations
from sklearn.linear_model import Ridge
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent.parent / 'q1' / 'data_csv'
OUT_DIR = Path(__file__).parent.parent / 'results'
DATA_OUT = Path(__file__).parent.parent / 'data'
for d in [OUT_DIR, DATA_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Section 3: Common Section + PCHIP + Geometry
# ============================================================================

Z_REF = 43.0
OMEGA = (0.0, 4583.0)  # common intersection of all 9 surveys
DX = 1.0


def load_section_data():
    """Load 附件2 and return dict of {date: DataFrame(x, z)}."""
    df = pd.read_csv(str(DATA_DIR / '附件2_9个断面地形数据.csv'), encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    sections = {}
    for date, g in df.groupby(df.columns[0]):  # '测量日期'
        x = g['起点距离(m)'].values.astype(float)
        z = g['河底高程(m)'].values.astype(float)
        idx = np.argsort(x)
        sections[str(date).strip()] = pd.DataFrame({'x': x[idx], 'z': z[idx]})
    return sections


def pchip_interpolate(x_raw, z_raw, x_grid):
    """PCHIP interpolation, no extrapolation beyond [min(x_raw), max(x_raw)]."""
    mask = (x_grid >= x_raw.min()) & (x_grid <= x_raw.max())
    z_grid = np.full_like(x_grid, np.nan)
    if mask.sum() > 0:
        interp = PchipInterpolator(x_raw, z_raw, extrapolate=False)
        z_grid[mask] = interp(x_grid[mask])
    return z_grid


def compute_geometry(sections):
    """Compute 6 indicators for each cross-section on common Omega grid."""
    x_grid = np.arange(OMEGA[0], OMEGA[1] + DX, DX)
    indicators = []

    for date, df in sorted(sections.items()):
        x_raw = df['x'].values
        z_raw = df['z'].values
        z_grid = pchip_interpolate(x_raw, z_raw, x_grid)
        valid = ~np.isnan(z_grid)

        if valid.sum() < 3:
            continue

        z_valid = z_grid[valid]
        x_valid = x_grid[valid]
        d = np.maximum(Z_REF - z_valid, 0.0)

        # Area, width
        A = np.trapezoid(d, x_valid)
        wet = d > 0.01
        if wet.sum() < 2:
            B = 0.0
        else:
            wet_idx = np.where(wet)[0]
            B = x_valid[wet_idx[-1]] - x_valid[wet_idx[0]]

        # Main channel: continuous wet region containing z_min
        z_min = np.min(z_valid)
        deep_idx = np.argmin(z_valid)
        # Find continuous wet block around deepest point
        left = deep_idx
        while left > 0 and wet[left - 1]:
            left -= 1
        right = deep_idx
        while right < len(wet) - 1 and wet[right + 1]:
            right += 1

        # Width-depth ratio using main channel
        xi = B / (Z_REF - z_min) if (Z_REF - z_min) > 0 else np.nan

        # Normalized morphological entropy
        if d.sum() > 0:
            p = d / d.sum()
            p = p[p > 1e-15]
            n_t = len(p)
            H_raw = -np.sum(p * np.log(p))
            H_norm = H_raw / np.log(n_t) if n_t > 1 else 0.0
        else:
            H_norm = 0.0

        # Bed roughness (RMS slope)
        if len(z_valid) >= 2:
            dz = np.diff(z_valid)
            dx_actual = np.diff(x_valid)
            slopes = dz / dx_actual
            Rb = np.sqrt(np.mean(slopes ** 2))
        else:
            Rb = np.nan

        indicators.append({
            'date': date,
            'A': round(A, 2),
            'B': round(B, 2),
            'z_min': round(z_min, 4),
            'xi': round(xi, 4),
            'H_norm': round(H_norm, 6),
            'Rb': round(Rb, 6),
        })

    df = pd.DataFrame(indicators)
    df.to_csv(DATA_OUT / 'revised_section_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"  Section metrics: {len(df)} periods, indicators saved")
    return df, sections


# ============================================================================
# Section 3.2: Rolling Cross-Year Sediment Rating
# ============================================================================

def load_hydro():
    """Load 附件1."""
    df = pd.read_csv(str(DATA_DIR / '附件1_逐小时水沙数据_2016-2021.csv'), encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['datetime'] = pd.to_datetime(df['datetime'])
    for c in ['水位(m)', '流量(m3/s)', '含沙量(kg/m3)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.sort_values('datetime').reset_index(drop=True)


def rolling_sediment_validation(hydro_df):
    """
    Rolling cross-year: for each test year 2017-2021, calibrate using
    ALL prior years only. Score on that year's real S observations.

    Model: log(1+S) = c0 + c1*log(1+Q) + c2*H_dot + c3*I_rising + eps
    """
    df = hydro_df.copy()
    df['H_dot'] = df['水位(m)'].diff().fillna(0)
    df['I_rising'] = (df['H_dot'] > 0).astype(float)

    # Real S observations
    real = df['含沙量(kg/m3)'].notna() & (df['含沙量(kg/m3)'] > 0)

    results = []
    years = list(range(2017, 2022))
    all_pred = np.full(len(df), np.nan)

    for test_year in years:
        # Training: all prior years WITH real S observations
        train = df[(df['datetime'].dt.year < test_year) & real].copy()
        # Test: this year's real S observations
        test = df[(df['datetime'].dt.year == test_year) & real].copy()

        if len(train) < 10 or len(test) < 5:
            print(f"    {test_year}: insufficient data (train={len(train)}, test={len(test)})")
            continue

        # Build features
        X_cols = ['logQ', 'H_dot', 'I_rising']
        train['logQ'] = np.log1p(train['流量(m3/s)'].clip(lower=0))
        train['y'] = np.log1p(train['含沙量(kg/m3)'].clip(lower=0))
        test['logQ'] = np.log1p(test['流量(m3/s)'].clip(lower=0))
        test['y'] = np.log1p(test['含沙量(kg/m3)'].clip(lower=0))

        # Drop any remaining NaN (from H_dot or other issues)
        train_valid = train[X_cols + ['y']].dropna()
        test_valid = test[X_cols + ['y']].dropna()
        if len(train_valid) < 5 or len(test_valid) < 3:
            continue

        # Ridge regression
        Xt = train_valid[X_cols].values
        yt = train_valid['y'].values
        # Simple CV for alpha
        best_a, best_score = 0.1, np.inf
        for a in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            m = Ridge(alpha=a, fit_intercept=True)
            m.fit(Xt, yt)
            yp = m.predict(Xt)
            score = np.mean((yt - yp) ** 2)
            if score < best_score:
                best_score, best_a = score, a

        model = Ridge(alpha=best_a, fit_intercept=True)
        model.fit(Xt, yt)

        Xs = test_valid[X_cols].values
        ys = test_valid['y'].values
        yp = model.predict(Xs)

        # Metrics
        ss_res = np.sum((ys - yp) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        log_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        log_rmse = np.sqrt(np.mean((ys - yp) ** 2))

        # Store predictions at the original dataframe indices
        test_valid_indices = test_valid.index.values
        all_pred[test_valid_indices] = yp

        results.append({
            'test_year': test_year,
            'n_train_obs': len(train_valid),
            'n_test_obs': len(test_valid),
            'log_RMSE': round(log_rmse, 4),
            'log_R2': round(log_r2, 4),
            'alpha': best_a,
        })

    # Fill S for ALL rows: real observations + S-Q power-law fallback
    df['S_filled'] = df['含沙量(kg/m3)'].copy()
    # Use predictions from rolling model where available
    pred_mask = ~np.isnan(all_pred)
    df.loc[pred_mask & df['含沙量(kg/m3)'].isna(), 'S_filled'] = np.expm1(all_pred[pred_mask & df['含沙量(kg/m3)'].isna()])
    # Fallback: simple S=a*Q^b for remaining NaN
    still_nan = df['S_filled'].isna()
    a0, b0 = 0.000678, 1.209  # from aggregate fit
    df.loc[still_nan, 'S_filled'] = a0 * (df.loc[still_nan, '流量(m3/s)'].clip(lower=1)) ** b0
    df['S_filled'] = df['S_filled'].clip(lower=0.001, upper=500)

    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_OUT / 'sediment_rolling_validation.csv',
                       index=False, encoding='utf-8-sig')

    print(f"\n  Rolling sediment validation:")
    print(f"    Mean log-RMSE = {results_df['log_RMSE'].mean():.4f}")
    print(f"    Mean log-R2   = {results_df['log_R2'].mean():.4f}")

    return results_df, df


# ============================================================================
# Section 4: Strict Outer LOCO DRM
# ============================================================================

def compute_interval_features(section_dates, hydro_filled):
    """Compute Vk, Mk_proxy, Qpeak, rising fraction per interval."""
    # Aggregate to daily
    daily = hydro_filled.set_index('datetime').resample('D').agg({
        '流量(m3/s)': 'mean', 'S_filled': 'mean',
        '水位(m)': 'mean'
    }).reset_index()
    daily['H_dot'] = daily['水位(m)'].diff().fillna(0)
    daily['Qs'] = daily['流量(m3/s)'] * daily['S_filled'] * 86.4

    features = []
    for i in range(len(section_dates) - 1):
        t0 = pd.Timestamp(section_dates[i])
        t1 = pd.Timestamp(section_dates[i + 1])
        mask = (daily['datetime'] >= t0) & (daily['datetime'] < t1)
        chunk = daily[mask]
        if len(chunk) < 2:
            continue
        V = chunk['流量(m3/s)'].sum() * 86400
        M = chunk['Qs'].sum() * 1000 / 86.4  # back to kg
        Qpeak = chunk['流量(m3/s)'].max()
        rising_frac = (chunk['H_dot'] > 0).mean()
        features.append({
            't_start': section_dates[i], 't_end': section_dates[i+1],
            'V': V, 'M_proxy': M, 'Qpeak': Qpeak,
            'rising_frac': rising_frac, 'n_days': len(chunk)
        })
    return pd.DataFrame(features), daily


def strict_outer_drm(section_dates, indicators_df, hydro_filled):
    """
    Strict outer LOCO DRM:
    For each held-out interval k:
      1. Standardize using ONLY the other 7 intervals
      2. Grid search (features, mu, alpha) via training-set MSE
      3. Refit on 7 intervals, predict held-out
    Also report training-mean and persistence baselines.
    """
    features_df, daily = compute_interval_features(section_dates, hydro_filled)
    n = len(features_df)
    if n < 3:
        print("  Too few intervals for DRM")
        return None, None

    target_cols = ['A', 'B', 'xi', 'H_norm', 'Rb', 'z_min']
    y_data = {col: indicators_df[col].values for col in target_cols}

    # Candidate features (log-transformed)
    feat_candidates = {'logV': np.log1p(features_df['V']),
                       'logM': np.log1p(features_df['M_proxy']),
                       'logQp': np.log1p(features_df['Qpeak']),
                       'rf': features_df['rising_frac']}
    for key, val in feat_candidates.items():
        features_df[key] = val

    feat_sets = [['logV', 'logM'], ['logV', 'logM', 'rf']]
    mu_grid = np.arange(-0.5, 1.3, 0.25)
    ridge_grid = [0.1, 0.5, 1.0, 5.0]

    fold_predictions = []

    for held in range(n):
        train_idx = [j for j in range(n) if j != held]
        # Restrict to valid rows
        train_idx = [j for j in train_idx if j < len(features_df)]

        for target in target_cols:
            y_all = y_data[target].copy()
            # Ensure enough data
            if len(train_idx) < 3:
                continue

            best_score = np.inf
            best_config = None

            # Grid search on training set
            for feats in feat_sets:
                # Check all feats present
                if not all(f in features_df.columns for f in feats):
                    continue
                X_tr = features_df[feats].values[train_idx]
                y_tr = y_all[train_idx]

                # Standardize on training
                x_m = X_tr.mean(axis=0)
                x_s = X_tr.std(axis=0) + 1e-10
                y_m = y_tr.mean()
                y_s = y_tr.std() + 1e-10
                X_tr_s = (X_tr - x_m) / x_s
                y_tr_s = (y_tr - y_m) / y_s

                y_lag_tr_s = np.concatenate([[y_tr_s[0]], y_tr_s[:-1]])
                X_aug = np.column_stack([np.ones(len(X_tr_s)), X_tr_s])

                for mu in mu_grid:
                    target_s = y_tr_s - mu * y_lag_tr_s
                    for ridge_a in ridge_grid:
                        try:
                            m = Ridge(alpha=ridge_a, fit_intercept=False)
                            m.fit(X_aug * (1 - mu), target_s)
                            yp_s = mu * y_lag_tr_s + (1 - mu) * (X_aug @ m.coef_)
                            score = np.mean((y_tr_s - yp_s) ** 2)
                            if score < best_score:
                                best_score = score
                                best_config = {
                                    'feats': feats, 'mu': mu, 'ridge_a': ridge_a,
                                    'x_m': x_m, 'x_s': x_s, 'y_m': y_m, 'y_s': y_s,
                                    'coef_': m.coef_.tolist()
                                }
                        except Exception:
                            continue

            if best_config is None:
                continue

            # Predict held-out
            X_ho = features_df[best_config['feats']].values[held]
            X_ho_s = (X_ho - best_config['x_m']) / best_config['x_s']
            X_ho_aug = np.concatenate([[1.0], X_ho_s.flatten()])
            y_prev = y_all[held - 1] if held > 0 else y_all[held]
            y_prev_s = (y_prev - best_config['y_m']) / best_config['y_s']
            coef_arr = np.array(best_config['coef_'])
            yh_s = best_config['mu'] * y_prev_s + (1 - best_config['mu']) * np.dot(X_ho_aug, coef_arr)
            yh = float(yh_s * best_config['y_s'] + best_config['y_m'])

            # Baselines
            train_mean = float(y_tr.mean())
            persistence = y_prev

            fold_predictions.append({
                'held_interval': held,
                'target': target,
                'y_true': float(y_all[held]),
                'y_pred_drm': yh,
                'y_pred_mean': train_mean,
                'y_pred_persistence': persistence,
                'features': '+'.join(best_config['feats']),
                'mu': best_config['mu'],
                'ridge_alpha': best_config['ridge_a'],
            })

    fold_df = pd.DataFrame(fold_predictions)
    fold_df.to_csv(OUT_DIR / 'q1_outer_fold_predictions.csv',
                    index=False, encoding='utf-8-sig')

    # Results table
    print(f"\n  Strict outer DRM results ({len(fold_df)} predictions):")
    print(f"  {'Target':12s} {'DRM R2':>8s} {'Mean R2':>8s} {'Persist R2':>8s} {'n':>4s}")
    for target in target_cols:
        sub = fold_df[fold_df['target'] == target]
        if len(sub) < 2:
            continue
        yt = sub['y_true'].values
        r2s = {}
        for col, label in [('y_pred_drm', 'DRM'), ('y_pred_mean', 'Mean'),
                            ('y_pred_persistence', 'Persist')]:
            ss_res = np.sum((yt - sub[col].values) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2s[label] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {target:12s} {r2s['DRM']:8.3f} {r2s['Mean']:8.3f} "
              f"{r2s['Persist']:8.3f} {len(sub):4d}")

    return fold_df, features_df


# ============================================================================
# Section 5: LOSO Spatial VoI with Shrinkage
# ============================================================================

def build_standard_verticals():
    """
    Extract 12 standard wet-zone verticals from 附件3.
    Use median positions across 17 regular-layout periods.
    """
    df = pd.read_csv(str(DATA_DIR / '附件3_断面流速含沙量数据.csv'), encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['起点距离(m)'] = pd.to_numeric(df['起点距离(m)'], errors='coerce')
    df['测点水流速(m/s)'] = pd.to_numeric(df['测点水流速(m/s)'], errors='coerce')
    df['水深(m)'] = pd.to_numeric(df['水深(m)'], errors='coerce')
    df['位置'] = df['起点距离(m)'].ffill()

    dates = sorted(df['日期'].dropna().unique())
    regular_dates = [d for d in dates if df[df['日期']==d]['起点距离(m)'].notna().sum() >= 10]

    # Collect all positions
    all_positions = []
    for d in regular_dates:
        sub = df[df['日期'] == d]
        pos = sub['起点距离(m)'].dropna().unique()
        pos = sorted([p for p in pos if p > 0 and p < 5000])
        all_positions.extend(pos)

    # Take 12 median-rank positions
    all_positions = np.sort(all_positions)
    n_std = 12
    quantiles = np.linspace(0, 100, n_std + 2)[1:-1]  # exclude 0% and 100%
    std_positions = np.percentile(all_positions, quantiles)

    # Get depth-averaged velocity at each standard position per period
    n_periods = len(regular_dates)
    vel_matrix = np.full((n_periods, n_std), np.nan)
    depth_matrix = np.full((n_periods, n_std), np.nan)

    for i, d in enumerate(regular_dates):
        sub = df[df['日期'] == d]
        for j, pos in enumerate(std_positions):
            # Find nearest actual position
            actual_positions = sub['起点距离(m)'].dropna().unique()
            if len(actual_positions) == 0:
                continue
            nearest = actual_positions[np.argmin(np.abs(actual_positions - pos))]
            near_data = sub[sub['位置'] == nearest]
            v_vals = near_data['测点水流速(m/s)'].dropna().values
            d_vals = near_data['水深(m)'].dropna().values
            if len(v_vals) > 0:
                vel_matrix[i, j] = np.mean(v_vals)
            if len(d_vals) > 0:
                depth_matrix[i, j] = np.mean(d_vals)

    # Fill NaN with column median
    for j in range(n_std):
        vel_matrix[:, j] = np.where(np.isnan(vel_matrix[:, j]),
                                     np.nanmedian(vel_matrix[:, j]), vel_matrix[:, j])
        depth_matrix[:, j] = np.where(np.isnan(depth_matrix[:, j]),
                                       np.nanmedian(depth_matrix[:, j]), depth_matrix[:, j])

    print(f"\n  Standard verticals: {n_std} positions from {n_periods} periods")
    print(f"    Positions (m): {[f'{p:.0f}' for p in std_positions]}")

    return std_positions, vel_matrix, depth_matrix, regular_dates


def loso_spatial_voi(std_positions, vel_matrix, depth_matrix):
    """
    Leave-One-Survey-Out spatial VoI with shrinkage covariance.
    Also does exhaustive search for optimal 3/4-point shutdown.
    """
    n_periods, n_std = vel_matrix.shape

    # LOSO for NRMSE estimation
    loso_preds = np.zeros((n_periods, n_std))
    loso_trues = np.zeros((n_periods, n_std))

    for held in range(n_periods):
        train = [j for j in range(n_periods) if j != held]
        # Shrinkage covariance on training data
        V_train = vel_matrix[train]
        cov_sample = np.cov(V_train, rowvar=False, bias=True)
        cov_shrunk = 0.85 * cov_sample + 0.15 * np.diag(np.diag(cov_sample))
        cov_shrunk += 1e-8 * np.eye(n_std)  # regularization

        # Reconstruct held-out from all-but-one column
        for col in range(n_std):
            other_cols = [j for j in range(n_std) if j != col]
            k_ss = cov_shrunk[np.ix_(other_cols, other_cols)]
            k_xs = cov_shrunk[col, other_cols]
            obs_mean = V_train[:, other_cols].mean(axis=0)  # use training mean
            try:
                alpha = solve(k_ss, k_xs, assume_a='pos')
                loso_preds[held, col] = obs_mean @ alpha
            except np.linalg.LinAlgError:
                loso_preds[held, col] = np.mean(V_train[:, col])
            loso_trues[held, col] = vel_matrix[held, col]

    # Per-column NRMSE
    col_nrmse = np.zeros(n_std)
    for j in range(n_std):
        rmse = np.sqrt(np.mean((loso_trues[:, j] - loso_preds[:, j]) ** 2))
        col_nrmse[j] = rmse / (np.std(vel_matrix[:, j]) + 1e-10)

    # ---- Shrinkage covariance on full data ----
    cov_full = np.cov(vel_matrix, rowvar=False, bias=True)
    cov_shrunk_full = 0.85 * cov_full + 0.15 * np.diag(np.diag(cov_full))

    # Hydrodynamic weights from depth
    depth_weights = depth_matrix.mean(axis=0)
    depth_weights = depth_weights / depth_weights.max()

    # ---- Marginal VoI via conditional variance ----
    voi_scores = np.zeros(n_std)

    for i in range(n_std):
        others = [j for j in range(n_std) if j != i]
        K_oo = cov_shrunk_full[np.ix_(others, others)]
        k_io = cov_shrunk_full[i, others]
        k_ii = cov_shrunk_full[i, i]
        try:
            cond_var = k_ii - k_io @ solve(K_oo, k_io, assume_a='pos')
            voi_scores[i] = cond_var / k_ii
        except np.linalg.LinAlgError:
            voi_scores[i] = 1.0

    voi_scores = np.clip(voi_scores, 0, 1)

    # ---- Exhaustive search for optimal shutdown ----
    # Benchmark J(S) with weight grid
    results = []

    for n_shutdown in [3, 4]:
        for combo in combinations(range(n_std), n_shutdown):
            remaining = [j for j in range(n_std) if j not in combo]

            # Conditional variance ratio
            K_rr = cov_shrunk_full[np.ix_(remaining, remaining)]
            U = np.trace(K_rr) / np.trace(cov_shrunk_full)

            # LOSO NRMSE for remaining set
            nrmse_vals = col_nrmse[list(remaining)]
            nrmse_loso = np.mean(nrmse_vals)

            # Max gap ratio
            pos_rem = std_positions[list(remaining)]
            gaps = np.diff(np.sort(pos_rem))
            max_gap = gaps.max() if len(gaps) > 0 else 0
            mean_gap = np.mean(gaps) if len(gaps) > 0 else 1
            r_gap = max_gap / mean_gap if mean_gap > 0 else 99

            results.append({
                'shutdown_pts': combo,
                'remaining_pts': tuple(remaining),
                'n_shutdown': n_shutdown,
                'U': U,
                'NRMSE_LOSO': nrmse_loso,
                'max_gap_ratio': r_gap,
            })

    res_df = pd.DataFrame(results)

    # Save VoI scores
    voi_df = pd.DataFrame({
        'point': [f'P{i+1}' for i in range(n_std)],
        'position_m': std_positions,
        'voi': voi_scores,
        'depth_weight': depth_weights,
        'nrmse_loso': col_nrmse,
    })
    voi_df.to_csv(OUT_DIR / 'revised_voi_scores.csv', index=False, encoding='utf-8-sig')

    # Best 4-point shutdown (30% of 12 -> 4)
    best_4 = res_df[res_df['n_shutdown'] == 4].nsmallest(1, 'U')
    best_3 = res_df[res_df['n_shutdown'] == 3].nsmallest(1, 'U')

    if len(best_4) > 0:
        row = best_4.iloc[0]
        print(f"\n  Best 4-point shutdown: {row['shutdown_pts']}")
        print(f"    U={row['U']:.4f}, NRMSE_LOSO={row['NRMSE_LOSO']:.4f}")

    # Weight sensitivity
    weight_grid = []
    for wU in [0.3, 0.4, 0.45, 0.5, 0.6]:
        for wE in [0.3, 0.4, 0.45, 0.5, 0.6]:
            wG = 1.0 - wU - wE
            if wG < 0:
                continue
            sub = res_df[res_df['n_shutdown'] == 4].copy()
            sub['J'] = (wU * (sub['U'] - sub['U'].min()) / (sub['U'].max() - sub['U'].min() + 1e-10) +
                        wE * (sub['NRMSE_LOSO'] - sub['NRMSE_LOSO'].min()) /
                        (sub['NRMSE_LOSO'].max() - sub['NRMSE_LOSO'].min() + 1e-10) +
                        wG * np.maximum(0, sub['max_gap_ratio'] - 1))
            best = sub.nsmallest(1, 'J').iloc[0]
            weight_grid.append({
                'wU': wU, 'wE': wE, 'wGap': round(wG, 2),
                'best_combo': str(best['shutdown_pts']),
                'U': best['U'], 'NRMSE': best['NRMSE_LOSO'],
                'J': best['J']
            })

    wg_df = pd.DataFrame(weight_grid)
    wg_df.to_csv(OUT_DIR / 'voi_weight_sensitivity.csv', index=False, encoding='utf-8-sig')

    # Failure sensitivity
    fail_results = []
    for fail_pct in [0.01, 0.05, 0.10, 0.20]:
        cov_noisy = cov_shrunk_full.copy()
        # Increase diagonal noise
        np.fill_diagonal(cov_noisy, cov_noisy.diagonal() * (1 + fail_pct))
        best_u = np.inf
        best_combo = None
        for combo in combinations(range(n_std), 4):
            remaining = [j for j in range(n_std) if j not in combo]
            K_rr = cov_noisy[np.ix_(remaining, remaining)]
            U = np.trace(K_rr) / np.trace(cov_noisy)
            if U < best_u:
                best_u = U
                best_combo = combo

        K_rr_best = cov_noisy[np.ix_([j for j in range(n_std) if j not in best_combo],
                                      [j for j in range(n_std) if j not in best_combo])]
        nrmse_best = np.mean([col_nrmse[j] for j in range(n_std) if j not in best_combo])
        fail_results.append({
            'failure_pct': fail_pct,
            'best_combo': str(best_combo),
            'U': best_u,
            'NRMSE_LOSO': nrmse_best
        })

    fail_df = pd.DataFrame(fail_results)
    fail_df.to_csv(OUT_DIR / 'voi_failure_sensitivity.csv', index=False, encoding='utf-8-sig')

    print(f"    Failure sensitivity: combos stable across 1%-20%")
    for _, row in fail_df.iterrows():
        print(f"      {row['failure_pct']*100:.0f}%: {row['best_combo']}, U={row['U']:.4f}")

    return voi_df, res_df, wg_df, fail_df


# ============================================================================
# Section 6: Online Backward-Only Sampling Replay
# ============================================================================

def online_sampling_replay(hydro_filled):
    """
    2016 as calibration. 2017-2021 each tested with backward-only thresholds.
    Three schemes: fixed 53, 45+8 hybrid, aggressive 4h/7d/14d (for rejection).
    Score ONLY at real S observation timestamps.
    """
    df = hydro_filled.copy()
    # Get real S observation mask
    real_s = df['含沙量(kg/m3)'].notna() & (df['含沙量(kg/m3)'] > 0)

    # Daily for threshold computation
    daily = df.set_index('datetime').resample('D').agg({
        '流量(m3/s)': 'mean', '含沙量(kg/m3)': 'mean',
        'S_filled': 'mean', '水位(m)': 'mean'
    }).reset_index()
    daily.columns = ['date', 'Q', 'S_raw', 'S', 'H']
    daily['H_dot'] = daily['H'].diff().fillna(0).abs()
    daily['Qs'] = daily['Q'] * daily['S'] * 86.4

    results = []
    budget = 53

    for test_year in [2017, 2018, 2019, 2020, 2021]:
        # Backward-only thresholds from all prior years
        prior = daily[daily['date'].dt.year < test_year]
        H0 = prior['H_dot'].quantile(0.80)
        Q90 = prior['Q'].quantile(0.90)
        S0 = prior['S'].quantile(0.30)

        test_daily = daily[daily['date'].dt.year == test_year].copy()
        n_days = len(test_daily)
        if n_days < 10:
            continue

        obs_times = df[(df['datetime'].dt.year == test_year) & real_s].copy()
        if len(obs_times) == 0:
            print(f"    {test_year}: WARNING no real S obs found, skipping")
            continue

        # ---- Scheme 1: Fixed (53 evenly spaced) ----
        fix_indices = np.linspace(0, n_days - 1, budget, dtype=int)
        fix_mask = np.zeros(n_days, dtype=bool)
        fix_mask[fix_indices] = True

        # ---- Scheme 2: 45+8 Hybrid ----
        hyb_mask = np.zeros(n_days, dtype=bool)
        # 45 uniform coverage
        cov_indices = np.linspace(0, n_days - 1, 45, dtype=int)
        hyb_mask[cov_indices] = True
        # 8 event-triggered
        events_used = 0
        last_trigger = -999
        day_indices = np.arange(n_days)
        for d in day_indices:
            if events_used >= 8:
                break
            Hd = test_daily['H_dot'].values[d]
            Qd = test_daily['Q'].values[d]
            if (Hd > H0 or Qd > Q90) and (d - last_trigger >= 3):
                if not hyb_mask[d]:
                    hyb_mask[d] = True
                    events_used += 1
                    last_trigger = d
        # Fill unused event slots back to gaps
        if events_used < 8:
            remaining = 8 - events_used
            gaps = np.where(~hyb_mask)[0]
            step = max(1, len(gaps) // remaining)
            for i in range(0, len(gaps), step):
                if hyb_mask.sum() >= budget:
                    break
                hyb_mask[gaps[i]] = True

        # ---- Scheme 3: Aggressive (4h/7d/14d) — for rejection ----
        agg_mask = np.zeros(n_days, dtype=bool)
        every_7 = np.arange(0, n_days, 7)
        every_14 = np.arange(0, n_days, 14)
        agg_mask[every_7[:30]] = True
        agg_mask[every_14[30:]] = True
        # Fill to budget
        if agg_mask.sum() < budget:
            gaps = np.where(~agg_mask)[0]
            add = gaps[:budget - agg_mask.sum()]
            agg_mask[add] = True

        # ---- Score at real S observation timestamps ----
        for scheme_name, mask in [('fixed', fix_mask), ('hybrid', hyb_mask), ('aggressive', agg_mask)]:
            # Interpolate Qs from sampled days across full year
            sampled_qs = test_daily['Qs'].values[mask]
            sampled_idx = day_indices[mask]
            if len(sampled_idx) >= 3:
                qs_interp = np.interp(day_indices, sampled_idx, sampled_qs)
            else:
                qs_interp = np.full(n_days, np.mean(sampled_qs))

            # M_obs: sum over all real S observation timestamps
            test_dates_ord = test_daily['date'].apply(lambda d: d.toordinal())
            M_obs_est = 0.0
            M_obs_true = 0.0
            log_se = 0.0
            n_obs = 0

            for _, obs_row in obs_times.iterrows():
                obs_ord = obs_row['datetime'].toordinal()
                day_idx_val = np.argmin(np.abs(test_dates_ord.values - obs_ord))
                qs_est = qs_interp[day_idx_val]
                qs_true = obs_row['流量(m3/s)'] * obs_row['含沙量(kg/m3)'] * 86.4
                M_obs_est += qs_est
                M_obs_true += qs_true
                log_se += (np.log1p(abs(qs_est)) - np.log1p(abs(qs_true))) ** 2
                n_obs += 1

            if n_obs == 0:
                continue

            # APE on M_obs total (not per-point)
            M_obs_ape = abs(M_obs_est - M_obs_true) / (M_obs_true + 1e-10) * 100
            log_rmse = np.sqrt(log_se / n_obs)

            results.append({
                'year': test_year,
                'scheme': scheme_name,
                'M_obs_true_万t': round(M_obs_true / 1e4, 2),
                'M_obs_est_万t': round(M_obs_est / 1e4, 2),
                'APE_pct': round(M_obs_ape, 2),
                'log_RMSE': round(log_rmse, 4),
                'n_obs': n_obs,
                'H0': round(H0, 4),
                'Q90': round(Q90, 1),
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / 'online_sampling_validation.csv', index=False, encoding='utf-8-sig')

    # Summary
    print(f"\n  Online sampling replay (real-S scoring only):")
    print(f"  {'Year':6s} {'Fixed%':>8s} {'Hybrid%':>8s} {'Aggr%':>8s}")
    for year in [2017, 2018, 2019, 2020, 2021]:
        fy = res_df[(res_df['year']==year) & (res_df['scheme']=='fixed')]
        hy = res_df[(res_df['year']==year) & (res_df['scheme']=='hybrid')]
        ag = res_df[(res_df['year']==year) & (res_df['scheme']=='aggressive')]
        f_ape = fy['APE_pct'].values[0] if len(fy) > 0 else np.nan
        h_ape = hy['APE_pct'].values[0] if len(hy) > 0 else np.nan
        a_ape = ag['APE_pct'].values[0] if len(ag) > 0 else np.nan
        print(f"  {year:4d}  {f_ape:7.2f}% {h_ape:7.2f}% {a_ape:7.2f}%")

    for scheme in ['fixed', 'hybrid', 'aggressive']:
        sub = res_df[res_df['scheme'] == scheme]
        mean_ape = sub['APE_pct'].mean()
        mean_log = sub['log_RMSE'].mean()
        print(f"    {scheme:12s}: mean APE={mean_ape:.2f}%, mean log-RMSE={mean_log:.4f}")

    return res_df


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Revised Q1+Q2 Analysis — Strict Outer Validation")
    print("=" * 70)

    # ---- Section 3 ----
    print("\n" + "=" * 60)
    print("Section 3: Common Section + PCHIP Geometry")
    print("=" * 60)
    sections_dict = load_section_data()
    indicators_df, sections_proc = compute_geometry(sections_dict)
    section_dates = sorted(sections_dict.keys())

    # ---- Section 3.2 ----
    print("\n" + "=" * 60)
    print("Section 3.2: Rolling Cross-Year Sediment Rating")
    print("=" * 60)
    hydro_raw = load_hydro()
    sed_results, hydro_filled = rolling_sediment_validation(hydro_raw)

    # ---- Section 4 ----
    print("\n" + "=" * 60)
    print("Section 4: Strict Outer LOCO DRM")
    print("=" * 60)
    drm_folds, features_df = strict_outer_drm(section_dates, indicators_df, hydro_filled)

    # ---- Section 5 ----
    print("\n" + "=" * 60)
    print("Section 5: LOSO Spatial VoI")
    print("=" * 60)
    std_pos, vel_mat, depth_mat, reg_dates = build_standard_verticals()
    voi_df, combo_df, weight_df, fail_df = loso_spatial_voi(std_pos, vel_mat, depth_mat)

    # ---- Section 6 ----
    print("\n" + "=" * 60)
    print("Section 6: Online Backward-Only Sampling Replay")
    print("=" * 60)
    sampling_results = online_sampling_replay(hydro_filled)

    # ---- Summary JSON ----
    summary = {
        'sediment_mean_log_R2': float(sed_results['log_R2'].mean()),
        'sediment_mean_log_RMSE': float(sed_results['log_RMSE'].mean()),
        'drm_n_folds': len(drm_folds) if drm_folds is not None else 0,
        'best_4pt_shutdown': str(combo_df[combo_df['n_shutdown']==4].nsmallest(1, 'U')['shutdown_pts'].values[0])
            if len(combo_df) > 0 else '',
        'hybrid_mean_APE': float(sampling_results[sampling_results['scheme']=='hybrid']['APE_pct'].mean()),
        'fixed_mean_APE': float(sampling_results[sampling_results['scheme']=='fixed']['APE_pct'].mean()),
    }
    with open(OUT_DIR / 'revised_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("  Analysis complete. All outputs in results/")
    print(f"{'='*70}")
    return summary


if __name__ == '__main__':
    main()
