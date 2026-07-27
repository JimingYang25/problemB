"""
Module 2: Geometric Indicators Calculation (v4 — 5 indicators)
Calculate for each cross-section:
  - A: Cross-sectional area below reference level
  - B: Water surface width
  - xi: Width-depth ratio (B^2/A)
  - H: Morphological entropy
  - z_min: Thalweg elevation (minimum bed elevation)

Reference water level Z_ref = 43m (fixed per README)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Reference water level (1985 National Elevation Datum)
Z_REF = 43.0  # meters


def calculate_cross_sectional_area(
    x: np.ndarray,
    z: np.ndarray,
    z_ref: float = Z_REF
) -> float:
    """
    Calculate cross-sectional area below reference water level.

    A = ∫ (Z_ref - z(x)) dx  for z(x) < Z_ref

    Uses trapezoidal integration.
    """
    # Only consider points below water level
    z_clipped = np.minimum(z, z_ref)
    depth = z_ref - z_clipped  # water depth at each point
    area = np.trapezoid(depth, x)
    return area


def calculate_surface_width(
    x: np.ndarray,
    z: np.ndarray,
    z_ref: float = Z_REF
) -> float:
    """
    Calculate water surface width.

    B = ∫ 1_{z(x) < Z_ref} dx

    Approximate: sum of x-intervals where z < Z_ref
    """
    below = z < z_ref
    width = 0.0

    for i in range(len(below) - 1):
        if below[i] and below[i + 1]:
            width += x[i + 1] - x[i]
        elif below[i] and not below[i + 1]:
            # Linear interpolation to find where z crosses Z_ref
            frac = (z_ref - z[i]) / (z[i + 1] - z[i]) if z[i + 1] != z[i] else 1.0
            width += frac * (x[i + 1] - x[i])
        elif not below[i] and below[i + 1]:
            frac = (z_ref - z[i]) / (z[i + 1] - z[i]) if z[i + 1] != z[i] else 0.0
            width += (1.0 - frac) * (x[i + 1] - x[i])

    return width


def calculate_morphological_entropy(
    x: np.ndarray,
    z: np.ndarray,
    z_ref: float = Z_REF,
    n_bins: int = 20
) -> float:
    """
    Calculate morphological entropy of the cross-section.

    H = -Σ p_j * ln(p_j)

    where p_j is the proportion of cross-sectional area in vertical bin j.
    Higher H → more irregular (rough/uneven) cross-section shape.

    The entropy is computed by dividing the submerged portion into
    vertical bins and computing the area fraction in each bin.
    """
    z_clipped = np.minimum(z, z_ref)
    depth = z_ref - z_clipped

    # Divide depth range into bins
    max_depth = depth.max()
    if max_depth <= 0:
        return 0.0

    bin_edges = np.linspace(0, max_depth, n_bins + 1)

    # Calculate area in each depth bin
    bin_areas = np.zeros(n_bins)

    for i in range(len(x) - 1):
        d1, d2 = depth[i], depth[i + 1]
        dx = x[i + 1] - x[i]

        for j in range(n_bins):
            lo, hi = bin_edges[j], bin_edges[j + 1]

            # Fraction of this segment in this bin
            # Average depth in this segment
            d_start = d1
            d_end = d2

            # Clip segment to bin boundaries
            seg_lo = max(lo, min(d_start, d_end))
            seg_hi = min(hi, max(d_start, d_end))

            if seg_hi > seg_lo:
                # Trapezoidal area in this bin
                frac = (seg_hi - seg_lo) / abs(d_end - d_start + 1e-10)
                local_dx = frac * dx
                avg_d = (seg_lo + seg_hi) / 2
                bin_areas[j] += local_dx * avg_d

    total_area = bin_areas.sum()
    if total_area <= 0:
        return 0.0

    p = bin_areas / total_area
    p = p[p > 1e-15]  # Avoid log(0)

    entropy = -np.sum(p * np.log(p))
    return entropy


def calculate_all_indicators(
    sections: dict,
    z_ref: float = Z_REF
) -> pd.DataFrame:
    """
    Calculate A, B, ξ, H for all cross-sections.

    Args:
        sections: {date_str: DataFrame with '起点距离(m)' and '河底高程(m)'}
        z_ref: Reference water level

    Returns:
        DataFrame with columns: 测量日期, A, B, ξ, H
    """
    results = []

    for date in sorted(sections.keys()):
        df = sections[date]
        x = df['起点距离(m)'].values.astype(float)
        z = df['河底高程(m)'].values.astype(float)

        A = calculate_cross_sectional_area(x, z, z_ref)
        B = calculate_surface_width(x, z, z_ref)
        xi = B * B / A if A > 0 else float('inf')
        H = calculate_morphological_entropy(x, z, z_ref)
        z_min = np.min(z)  # thalweg elevation

        results.append({
            '测量日期': date,
            'A_面积(m^2)': round(A, 2),
            'B_水面宽(m)': round(B, 2),
            'xi_宽深比': round(xi, 4),
            'H_形态熵(nats)': round(H, 6),
            'z_min_深泓(m)': round(z_min, 4)
        })

        print(f"  [{date}] A={A:.1f} m^2, B={B:.1f} m, xi={xi:.2f}, "
              f"H={H:.4f} nats, z_min={z_min:.2f} m")

    df = pd.DataFrame(results)
    return df


def calculate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate delta values for all 5 indicators between adjacent cross-sections.

    Returns DataFrame with 8 rows.
    """
    deltas = []
    cols = ['A_面积(m^2)', 'B_水面宽(m)', 'xi_宽深比',
            'H_形态熵(nats)', 'z_min_深泓(m)']
    delta_names = ['dA', 'dB', 'dxi', 'dH', 'dz_min']

    for i in range(len(df) - 1):
        row = {
            '起始日期': df.iloc[i]['测量日期'],
            '结束日期': df.iloc[i + 1]['测量日期']
        }
        for col, dname in zip(cols, delta_names):
            row[dname] = round(df.iloc[i + 1][col] - df.iloc[i][col], 6)
        deltas.append(row)

    return pd.DataFrame(deltas)


def main():
    from preprocess import load_cross_sections, process_all_sections

    data_dir = Path(__file__).parent.parent / 'data_csv'
    output_dir = Path(__file__).parent.parent / 'data'

    print("=" * 60)
    print("Module 2: Geometric Indicators Calculation")
    print("=" * 60)
    print(f"\nReference water level Z_ref = {Z_REF}m")

    # Load and process data
    filepath = data_dir / '附件2_9个断面地形数据.csv'
    raw_sections = load_cross_sections(str(filepath))
    sections = process_all_sections(raw_sections, dx=1.0)

    # Calculate indicators
    print("\n--- Calculating Geometric Indicators ---")
    indicators_df = calculate_all_indicators(sections)

    # Calculate deltas
    deltas_df = calculate_deltas(indicators_df)

    # Save outputs
    indicators_df.to_csv(
        output_dir / 'geometry_indicators.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"\nSaved indicators to {output_dir / 'geometry_indicators.csv'}")

    deltas_df.to_csv(
        output_dir / 'geometry_deltas.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"Saved deltas to {output_dir / 'geometry_deltas.csv'}")

    return indicators_df, deltas_df, sections


if __name__ == '__main__':
    main()
