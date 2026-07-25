"""
Module 5: Steady-State Equation Fitting (Logistic ODE Model)
Fit logistic growth model to cross-section area evolution over time:

  A(t) = A_eq / (1 + (A_eq/A_0 - 1) · exp(-r·t))

Similarly fit exponential decay for width-depth ratio and morphological entropy:
  ξ(t) = ξ_eq + (ξ_0 - ξ_eq) · exp(-r_ξ · t)
  H(t) = H_eq + (H_0 - H_eq) · exp(-r_H · t)

Stability analysis:
  f'(A_eq) = -r < 0 → Stable equilibrium (attractor)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit


def logistic_growth(t: np.ndarray, A_eq: float, r: float, A0: float) -> np.ndarray:
    """
    Logistic growth function.

    A(t) = A_eq / (1 + (A_eq/A0 - 1) * exp(-r * t))
    """
    epsilon = 1e-10
    ratio = A_eq / (A0 + epsilon)
    if ratio <= 0:
        ratio = 0.01  # avoid invalid values
    return A_eq / (1.0 + (ratio - 1.0) * np.exp(-r * t))


def exponential_decay(t: np.ndarray, X_eq: float, r: float, X0: float) -> np.ndarray:
    """
    Exponential decay/approach to equilibrium.

    X(t) = X_eq + (X0 - X_eq) * exp(-r * t)
    """
    return X_eq + (X0 - X_eq) * np.exp(-r * t)


def fit_logistic_model(
    t_days: np.ndarray,
    A_values: np.ndarray
) -> dict:
    """
    Fit logistic growth model to area evolution data.

    Args:
        t_days: Time in days since first measurement
        A_values: Cross-sectional area at each time point

    Returns:
        dict with fitted parameters and diagnostics
    """
    # Initial guesses
    A0_guess = A_values[0]
    # Use mean as equilibrium guess (more stable than last value)
    A_eq_guess = np.mean(A_values)
    r_guess = 0.001  # Slow adjustment rate

    try:
        popt, pcov = curve_fit(
            logistic_growth, t_days, A_values,
            p0=[A_eq_guess, r_guess, A0_guess],
            bounds=([0.5 * A_values.min(), 0, 0.5 * A_values.min()],
                    [2.0 * A_values.max(), 1.0, 2.0 * A_values.max()]),
            maxfev=100000
        )
        A_eq, r, A0 = popt
        perr = np.sqrt(np.diag(pcov))
        success = True
    except (RuntimeError, ValueError) as e:
        print(f"  WARNING: Logistic fit failed: {e}")
        print(f"  Falling back to manual estimates")
        A_eq = A_values[-1]
        r = 0.001
        A0 = A_values[0]
        perr = [np.nan, np.nan, np.nan]
        success = False

    # Predicted values
    A_pred = logistic_growth(t_days, A_eq, r, A0)

    # R²
    ss_res = np.sum((A_values - A_pred) ** 2)
    ss_tot = np.sum((A_values - A_values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Stability: f'(A_eq) = -r < 0 → stable
    is_stable = r > 0

    return {
        'A_eq': A_eq,
        'r': r,
        'A0': A0,
        'r2': r2,
        'is_stable': is_stable,
        'success': success,
        'se_A_eq': perr[0],
        'se_r': perr[1],
        'se_A0': perr[2],
        'predicted': A_pred.tolist(),
        'stability': '稳定平衡点 (Stable equilibrium)' if is_stable
                     else '不稳定 (Unstable)'
    }


def fit_exponential_model(
    t_days: np.ndarray,
    X_values: np.ndarray,
    variable_name: str
) -> dict:
    """
    Fit exponential decay to equilibrium model.

    X(t) = X_eq + (X0 - X_eq) * exp(-r * t)
    """
    # Initial guesses
    X0_guess = X_values[0]
    X_eq_guess = X_values[-1]
    r_guess = 0.001

    try:
        popt, pcov = curve_fit(
            exponential_decay, t_days, X_values,
            p0=[X_eq_guess, r_guess, X0_guess],
            bounds=([0.5 * X_values.min(), 0, 0.5 * X_values.min()],
                    [2.0 * X_values.max(), 1.0, 2.0 * X_values.max()]),
            maxfev=100000
        )
        X_eq, r, X0 = popt
        perr = np.sqrt(np.diag(pcov))
        success = True
    except (RuntimeError, ValueError) as e:
        print(f"  WARNING: Exponential fit for {variable_name} failed: {e}")
        X_eq = X_values[-1]
        r = 0.001
        X0 = X_values[0]
        perr = [np.nan, np.nan, np.nan]
        success = False

    X_pred = exponential_decay(t_days, X_eq, r, X0)

    ss_res = np.sum((X_values - X_pred) ** 2)
    ss_tot = np.sum((X_values - X_values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'X_eq': X_eq,
        'r': r,
        'X0': X0,
        'r2': r2,
        'success': success,
        'predicted': X_pred.tolist(),
        'converges_to': f'{X_eq:.4f} (r={r:.6f}, sign: {"→eq" if r>0 else "→diverge"})'
    }


def fit_all_steady_state(indicators_df: pd.DataFrame) -> dict:
    """
    Fit steady-state models for A(t), ξ(t), H(t).

    Returns:
        dict with all fitted parameters
    """
    # Convert dates to days since first measurement
    dates = pd.to_datetime(indicators_df['测量日期'])
    t_days = (dates - dates.min()).dt.total_seconds().values / 86400.0

    A_values = indicators_df['A_面积(m²)'].values
    xi_values = indicators_df['ξ_宽深比'].values
    H_values = indicators_df['H_形态熵(nats)'].values

    print("\n" + "=" * 60)
    print("Module 5: Steady-State Equation Fitting")
    print("=" * 60)

    # Fit area logistic
    print("\n--- Area A(t): Logistic Growth ---")
    print(f"  Time span: {t_days[0]:.0f} to {t_days[-1]:.0f} days ({t_days[-1]/365:.1f} years)")
    print(f"  Input: t={t_days.round(1)}, A={A_values}")

    result_A = fit_logistic_model(t_days, A_values)

    print(f"  Fitted parameters:")
    print(f"    A_eq = {result_A['A_eq']:.1f} ± {result_A['se_A_eq']:.1f} m²")
    print(f"    r    = {result_A['r']:.6f} ± {result_A['se_r']:.6f} day⁻¹")
    print(f"    A_0  = {result_A['A0']:.1f} ± {result_A['se_A0']:.1f} m²")
    print(f"  R² = {result_A['r2']:.4f}")
    print(f"  Stability: {result_A['stability']}")

    # Fit ξ exponential decay
    print("\n--- Width-depth Ratio ξ(t): Exponential Approach ---")
    result_xi = fit_exponential_model(t_days, xi_values, 'ξ')
    print(f"  ξ_eq = {result_xi['X_eq']:.2f}, r = {result_xi['r']:.6f}, R² = {result_xi['r2']:.4f}")

    # Fit H exponential decay
    print("\n--- Morphological Entropy H(t): Exponential Approach ---")
    result_H = fit_exponential_model(t_days, H_values, 'H')
    print(f"  H_eq = {result_H['X_eq']:.4f}, r = {result_H['r']:.6f}, R² = {result_H['r2']:.4f}")

    # Summary
    print("\n--- Summary of Steady-State Predictions ---")
    print(f"  平衡面积 A_eq  = {result_A['A_eq']:.1f} m² (stable={result_A['is_stable']})")
    print(f"  平衡宽深比 ξ_eq = {result_xi['X_eq']:.2f}")
    print(f"  平衡形态熵 H_eq = {result_H['X_eq']:.4f}")

    # Conclusion
    print("\n--- Conclusion ---")
    if result_A['is_stable']:
        print("  ✅ The cross-section TENDS toward a STABLE EQUILIBRIUM.")
        print(f"     Equilibrium area ≈ {result_A['A_eq']:.0f} m²")
        print(f"     Adjustment rate r = {result_A['r']:.4f} day⁻¹")
        half_life = np.log(2) / result_A['r'] if result_A['r'] > 0 else float('inf')
        print(f"     Half-life ≈ {half_life:.0f} days ({half_life/365:.1f} years)")
    else:
        print("  ⚠️  Unstable: cross-section may undergo IRREVERSIBLE morphological collapse.")

    return {
        'A': result_A,
        'ξ': result_xi,
        'H': result_H,
        't_days': t_days.tolist(),
        'dates': indicators_df['测量日期'].tolist()
    }


def save_steady_state_results(results: dict, output_path: str):
    """Save steady-state parameters to CSV."""
    rows = [
        {
            '变量': '面积 A (m²)',
            '平衡值 X_eq': round(results['A']['A_eq'], 2),
            '调整速率 r (day⁻¹)': round(results['A']['r'], 6),
            '初始值 X0': round(results['A']['A0'], 2),
            'R²': round(results['A']['r2'], 4),
            '稳定?': results['A']['is_stable'],
            '模型': 'Logistic growth'
        },
        {
            '变量': '宽深比 ξ',
            '平衡值 X_eq': round(results['ξ']['X_eq'], 4),
            '调整速率 r (day⁻¹)': round(results['ξ']['r'], 6),
            '初始值 X0': round(results['ξ']['X0'], 4),
            'R²': round(results['ξ']['r2'], 4),
            '稳定?': results['ξ']['r'] > 0,
            '模型': 'Exponential approach'
        },
        {
            '变量': '形态熵 H (nats)',
            '平衡值 X_eq': round(results['H']['X_eq'], 6),
            '调整速率 r (day⁻¹)': round(results['H']['r'], 6),
            '初始值 X0': round(results['H']['X0'], 6),
            'R²': round(results['H']['r2'], 4),
            '稳定?': results['H']['r'] > 0,
            '模型': 'Exponential approach'
        }
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved steady-state results to {output_path}")


def main():
    from preprocess import load_cross_sections, process_all_sections
    from geometry import calculate_all_indicators

    data_dir = Path(__file__).parent.parent / 'data_csv'
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    sections_file = data_dir / '附件2_9个断面地形数据.csv'
    raw_sections = load_cross_sections(str(sections_file))
    sections = process_all_sections(raw_sections, dx=1.0)

    # Calculate indicators
    indicators_df = calculate_all_indicators(sections)

    # Fit steady-state models
    results = fit_all_steady_state(indicators_df)

    # Save
    save_steady_state_results(
        results,
        str(results_dir / 'steady_state_params.csv')
    )

    return results


if __name__ == '__main__':
    main()
