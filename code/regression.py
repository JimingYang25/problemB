"""
Module 4: Empirical Formula Fitting
Fit linear regression models relating cross-section geometry changes
to driving factors (cumulative runoff V, sediment transport M, peak discharge Qpeak).

Area evolution:
  ΔA = a1·ln(V) + a2·ln(M) + a3·Qpeak + a4 + ε

Width-depth ratio evolution:
  Δξ = b1·ln(V) + b2·ln(M) + b3·Qpeak + b4 + η

Interpretation:
  - a2 > 0: sediment accumulation increases area → scouring dominates
  - a2 < 0: sediment accumulation decreases area → deposition dominates
  - b2 > 0: sediment transport widens channel → wider/shallower
  - b2 < 0: sediment transport narrows channel → narrower/deeper
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression


def build_regression_data(
    deltas_df: pd.DataFrame,
    factors_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge geometry deltas with driving factors.

    Returns:
        DataFrame with columns: ΔA, Δξ, lnV, lnM, Qpeak
    """
    merged = deltas_df.merge(factors_df, on=['起始日期', '结束日期'], how='inner')

    # Create log-transformed features
    merged['lnV'] = np.log(merged['V_累计径流量(m³)'].replace(0, np.nan))
    merged['lnM'] = np.log(merged['M_累计输沙量(kg)'].replace(0, np.nan))

    # Replace -inf with NaN (from log(0 or near 0))
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['lnV', 'lnM', 'Q_peak(m³/s)', 'ΔA', 'Δξ']
    )

    return merged


def fit_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    target_name: str
) -> dict:
    """
    Fit linear regression with full statistics.

    Returns:
        dict with coefficients, p-values, R², etc.
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred
    n, k = X.shape[0], X.shape[1]

    # R² and adjusted R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2

    # Standard errors and p-values
    # Add intercept column
    X_with_ones = np.column_stack([np.ones(n), X])
    try:
        # Variance-covariance matrix
        sigma2 = ss_res / (n - k - 1) if n > k + 1 else ss_res / n
        cov_matrix = sigma2 * np.linalg.inv(X_with_ones.T @ X_with_ones)
        se = np.sqrt(np.diag(cov_matrix))
        t_stats = np.append(model.intercept_, model.coef_) / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=max(1, n - k - 1)))
    except np.linalg.LinAlgError:
        se = np.full(k + 1, np.nan)
        p_values = np.full(k + 1, np.nan)

    # Coefficients: [intercept, feature1, feature2, ...]
    coef_names = ['(Intercept)'] + feature_names
    coefs = np.array([model.intercept_] + list(model.coef_))

    return {
        'target': target_name,
        'n': n,
        'r2': r2,
        'r2_adj': r2_adj,
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'coefficients': {name: {'coef': round(c, 6), 'p_value': round(p, 6)}
                         for name, c, p in zip(coef_names, coefs, p_values)},
        'y_true': y.tolist(),
        'y_pred': y_pred.tolist(),
        'residuals': residuals.tolist()
    }


def fit_both_equations(merged: pd.DataFrame) -> tuple:
    """
    Fit both the ΔA and Δξ regression equations.
    """
    feature_names = ['ln(V)', 'ln(M)', 'Qpeak']
    X = merged[['lnV', 'lnM', 'Q_peak(m³/s)']].values
    y_A = merged['ΔA'].values
    y_xi = merged['Δξ'].values

    print(f"\n  Using {len(merged)} data points for regression")

    # Fit ΔA equation
    print("\n--- Area Evolution: ΔA = a1·ln(V) + a2·ln(M) + a3·Qpeak + a4 ---")
    result_A = fit_regression(X, y_A, feature_names, 'ΔA')

    print(f"  R² = {result_A['r2']:.4f}, R²_adj = {result_A['r2_adj']:.4f}, "
          f"RMSE = {result_A['rmse']:.2f}")
    for name, stats in result_A['coefficients'].items():
        sig = '***' if stats['p_value'] < 0.01 else ('**' if stats['p_value'] < 0.05 else
               ('*' if stats['p_value'] < 0.1 else ''))
        print(f"  {name:15s}: coef = {stats['coef']:10.4f}, p = {stats['p_value']:.4f} {sig}")

    # Interpret a2 (coefficient of lnM)
    a2 = result_A['coefficients']['ln(M)']['coef']
    if a2 > 0:
        print(f"  ▶ a2 = {a2:.4f} > 0: Sediment transport associates with AREA INCREASE (scouring)")
    else:
        print(f"  ▶ a2 = {a2:.4f} < 0: Sediment transport associates with AREA DECREASE (deposition)")

    # Fit Δξ equation
    print("\n--- Width-depth Ratio Evolution: Δξ = b1·ln(V) + b2·ln(M) + b3·Qpeak + b4 ---")
    result_xi = fit_regression(X, y_xi, feature_names, 'Δξ')

    print(f"  R² = {result_xi['r2']:.4f}, R²_adj = {result_xi['r2_adj']:.4f}, "
          f"RMSE = {result_xi['rmse']:.4f}")
    for name, stats in result_xi['coefficients'].items():
        sig = '***' if stats['p_value'] < 0.01 else ('**' if stats['p_value'] < 0.05 else
               ('*' if stats['p_value'] < 0.1 else ''))
        print(f"  {name:15s}: coef = {stats['coef']:10.6f}, p = {stats['p_value']:.4f} {sig}")

    # Interpret b2 (coefficient of lnM)
    b2 = result_xi['coefficients']['ln(M)']['coef']
    if b2 > 0:
        print(f"  ▶ b2 = {b2:.6f} > 0: Sediment transport → WIDER/SHALLOWER cross-section")
    else:
        print(f"  ▶ b2 = {b2:.6f} < 0: Sediment transport → NARROWER/DEEPER cross-section")

    return result_A, result_xi


def save_regression_results(result_A: dict, result_xi: dict, output_path: str):
    """Save regression coefficients to CSV."""
    rows = []

    for result in [result_A, result_xi]:
        target = result['target']
        for name, stats in result['coefficients'].items():
            rows.append({
                '目标变量': target,
                '参数': name,
                '系数': stats['coef'],
                'p值': stats['p_value'],
                'R²': result['r2'],
                'R²_adj': result['r2_adj'],
                'RMSE': result['rmse']
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved regression results to {output_path}")


def main():
    from preprocess import load_cross_sections, process_all_sections
    from geometry import calculate_all_indicators, calculate_deltas
    from driving_factors import load_hydro_data, extract_driving_factors

    data_dir = Path(__file__).parent.parent / 'data_csv'
    output_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Module 4: Empirical Formula Fitting")
    print("=" * 60)

    # ---- Rebuild data from modules 1-3 ----
    # Module 1
    print("\n[Step 1] Loading cross-section data...")
    sections_file = data_dir / '附件2_9个断面地形数据.csv'
    raw_sections = load_cross_sections(str(sections_file))
    sections = process_all_sections(raw_sections, dx=1.0)

    # Module 2
    print("\n[Step 2] Calculating geometric indicators...")
    indicators_df = calculate_all_indicators(sections)
    deltas_df = calculate_deltas(indicators_df)

    # Module 3
    print("\n[Step 3] Extracting driving factors...")
    hydro_file = data_dir / '附件1_逐小时水沙数据_2016-2021.csv'
    hydro_df = load_hydro_data(str(hydro_file))
    section_dates = sorted(sections.keys())
    date_pairs = [(section_dates[i], section_dates[i + 1])
                  for i in range(len(section_dates) - 1)]
    factors_df = extract_driving_factors(hydro_df, date_pairs, section_dates)

    # Module 4
    print("\n[Step 4] Building regression dataset...")
    merged = build_regression_data(deltas_df, factors_df)

    print("\n[Step 5] Fitting regression equations...")
    result_A, result_xi = fit_both_equations(merged)

    # Save
    save_regression_results(
        result_A, result_xi,
        str(results_dir / 'regression_coefficients.csv')
    )

    return result_A, result_xi, merged


if __name__ == '__main__':
    main()
