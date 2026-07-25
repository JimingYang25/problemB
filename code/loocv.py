"""
Module 6: Model Validation — Leave-One-Out Cross-Validation (LOOCV)
Validate the empirical formula by leaving out each cross-section sequentially,
training on the remaining 8, and predicting the held-out one.

Metrics:
  RMSE = sqrt( (1/n) * Σ (ŷ_i - y_i)² )
  MAPE = (1/n) * Σ |ŷ_i - y_i| / |y_i| × 100%
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression


def loocv_validation(
    merged: pd.DataFrame,
    n_sections: int = 9
) -> dict:
    """
    Perform Leave-One-Out Cross-Validation.

    Since we have 8 intervals (Δ data) between 9 sections, LOOCV leaves out
    one interval at a time. We train on 7 intervals and test on 1.

    Returns:
        dict with RMSE, MAPE, and per-fold predictions
    """
    X = merged[['lnV', 'lnM', 'Q_peak(m³/s)']].values
    y_A = merged['ΔA'].values
    y_xi = merged['Δξ'].values

    n = len(X)

    predictions = {
        'ΔA_pred': [],
        'ΔA_true': [],
        'Δξ_pred': [],
        'Δξ_true': [],
        'fold_info': []
    }

    print("\n" + "=" * 60)
    print("Module 6: Leave-One-Out Cross-Validation")
    print("=" * 60)

    for i in range(n):
        # Split
        train_idx = [j for j in range(n) if j != i]
        test_idx = i

        X_train = X[train_idx]
        y_A_train = y_A[train_idx]
        y_xi_train = y_xi[train_idx]
        X_test = X[test_idx].reshape(1, -1)
        y_A_test = y_A[test_idx]
        y_xi_test = y_xi[test_idx]

        # Train ΔA model
        model_A = LinearRegression()
        model_A.fit(X_train, y_A_train)
        y_A_pred = model_A.predict(X_test)[0]

        # Train Δξ model
        model_xi = LinearRegression()
        model_xi.fit(X_train, y_xi_train)
        y_xi_pred = model_xi.predict(X_test)[0]

        predictions['ΔA_pred'].append(y_A_pred)
        predictions['ΔA_true'].append(y_A_test)
        predictions['Δξ_pred'].append(y_xi_pred)
        predictions['Δξ_true'].append(y_xi_test)

        fold_name = f"Fold {i+1} ({merged.iloc[i]['起始日期']} → {merged.iloc[i]['结束日期']})"
        predictions['fold_info'].append(fold_name)

        print(f"  {fold_name}:")
        print(f"    ΔA:  true={y_A_test:.1f}, pred={y_A_pred:.1f}, "
              f"err={y_A_pred - y_A_test:.1f}")
        print(f"    Δξ:  true={y_xi_test:.4f}, pred={y_xi_pred:.4f}, "
              f"err={y_xi_pred - y_xi_test:.4f}")

    # Compute metrics
    y_A_true = np.array(predictions['ΔA_true'])
    y_A_pred = np.array(predictions['ΔA_pred'])
    y_xi_true = np.array(predictions['Δξ_true'])
    y_xi_pred = np.array(predictions['Δξ_pred'])

    metrics_A = compute_metrics(y_A_true, y_A_pred, 'ΔA')
    metrics_xi = compute_metrics(y_xi_true, y_xi_pred, 'Δξ')

    return metrics_A, metrics_xi, predictions


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Compute RMSE and MAPE."""
    errors = y_pred - y_true

    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE: avoid division by zero
    nonzero = np.abs(y_true) > 1e-10
    if nonzero.sum() > 0:
        mape = np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100
    else:
        mape = np.nan

    print(f"\n  --- {name} Validation Metrics ---")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAPE = {mape:.2f}%")

    return {
        'variable': name,
        'RMSE': round(rmse, 4),
        'MAPE(%)': round(mape, 2),
        'n_folds': len(y_true)
    }


def save_validation_results(
    metrics_A: dict,
    metrics_xi: dict,
    predictions: dict,
    output_path: str
):
    """Save validation results to CSV."""
    # Metrics summary
    metrics_df = pd.DataFrame([metrics_A, metrics_xi])
    metrics_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # Per-fold predictions
    fold_df = pd.DataFrame({
        'Fold': predictions['fold_info'],
        'ΔA_true': predictions['ΔA_true'],
        'ΔA_pred': predictions['ΔA_pred'],
        'ΔA_error': [p - t for p, t in zip(predictions['ΔA_pred'],
                                           predictions['ΔA_true'])],
        'Δξ_true': predictions['Δξ_true'],
        'Δξ_pred': predictions['Δξ_pred'],
        'Δξ_error': [p - t for p, t in zip(predictions['Δξ_pred'],
                                           predictions['Δξ_true'])]
    })

    fold_path = output_path.replace('.csv', '_per_fold.csv')
    fold_df.to_csv(fold_path, index=False, encoding='utf-8-sig')

    print(f"\nSaved validation metrics to {output_path}")
    print(f"Saved per-fold predictions to {fold_path}")


def main():
    from preprocess import load_cross_sections, process_all_sections
    from geometry import calculate_all_indicators, calculate_deltas
    from driving_factors import load_hydro_data, extract_driving_factors
    from regression import build_regression_data

    data_dir = Path(__file__).parent.parent / 'data_csv'
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild the full pipeline
    print("=" * 60)
    print("Building full pipeline for LOOCV...")
    print("=" * 60)

    # Module 1
    sections_file = data_dir / '附件2_9个断面地形数据.csv'
    raw_sections = load_cross_sections(str(sections_file))
    sections = process_all_sections(raw_sections, dx=1.0)

    # Module 2
    indicators_df = calculate_all_indicators(sections)
    deltas_df = calculate_deltas(indicators_df)

    # Module 3
    hydro_file = data_dir / '附件1_逐小时水沙数据_2016-2021.csv'
    hydro_df = load_hydro_data(str(hydro_file))
    section_dates = sorted(sections.keys())
    date_pairs = [(section_dates[i], section_dates[i + 1])
                  for i in range(len(section_dates) - 1)]
    factors_df = extract_driving_factors(hydro_df, date_pairs, section_dates)

    # Merge
    merged = build_regression_data(deltas_df, factors_df)

    # LOOCV
    metrics_A, metrics_xi, predictions = loocv_validation(merged, n_sections=9)

    # Save
    save_validation_results(
        metrics_A, metrics_xi, predictions,
        str(results_dir / 'validation_metrics.csv')
    )

    return metrics_A, metrics_xi, predictions


if __name__ == '__main__':
    main()
