"""
Module 1: Data Preprocessing
- Parse 附件2 cross-section measurement data
- Group by measurement date
- Interpolate to uniform 1m horizontal grid
- Output standardized cross-section data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d


def load_cross_sections(filepath: str) -> dict:
    """
    Load 附件2 and group by measurement date.

    Returns:
        dict: {date_str: DataFrame with columns ['起点距离(m)', '河底高程(m)']}
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    sections = {}
    for date, group in df.groupby('测量日期'):
        group = group.sort_values('起点距离(m)').reset_index(drop=True)
        sections[date] = group
        print(f"  [{date}] {len(group)} measurement points, x range: "
              f"[{group['起点距离(m)'].min():.0f}, {group['起点距离(m)'].max():.0f}]m")

    return sections


def interpolate_cross_section(
    df: pd.DataFrame,
    dx: float = 1.0
) -> pd.DataFrame:
    """
    Interpolate a single cross-section to a uniform horizontal grid.

    Args:
        df: Raw cross-section data with '起点距离(m)' and '河底高程(m)'
        dx: Grid spacing in meters (default 1m)

    Returns:
        DataFrame with columns ['起点距离(m)', '河底高程(m)'] on uniform grid
    """
    x_raw = df['起点距离(m)'].values.astype(float)
    z_raw = df['河底高程(m)'].values.astype(float)

    # Remove duplicates by taking mean at duplicate x positions
    unique_x, indices = np.unique(x_raw, return_inverse=True)
    z_unique = np.array([z_raw[indices == i].mean() for i in range(len(unique_x))])

    # Create uniform grid
    x_grid = np.arange(unique_x.min(), unique_x.max() + dx, dx)

    # Interpolate using cubic spline (linear fallback if too few points)
    if len(unique_x) >= 4:
        interp = interp1d(unique_x, z_unique, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
    else:
        interp = interp1d(unique_x, z_unique, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    z_grid = interp(x_grid)

    return pd.DataFrame({
        '起点距离(m)': x_grid,
        '河底高程(m)': z_grid
    })


def process_all_sections(
    sections: dict,
    dx: float = 1.0
) -> dict:
    """
    Interpolate all cross-sections to uniform grid.

    Returns:
        dict: {date_str: interpolated DataFrame}
    """
    interpolated = {}
    for date, df in sections.items():
        interp_df = interpolate_cross_section(df, dx=dx)
        interpolated[date] = interp_df
        print(f"  [{date}] Interpolated: {len(df)} → {len(interp_df)} points (dx={dx}m)")

    return interpolated


def save_processed_data(interpolated: dict, output_path: str):
    """Save all interpolated cross-sections to a single CSV."""
    rows = []
    for date, df in interpolated.items():
        for _, row in df.iterrows():
            rows.append({
                '测量日期': date,
                '起点距离(m)': row['起点距离(m)'],
                '河底高程(m)': row['河底高程(m)']
            })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved processed data to {output_path} ({len(result)} rows)")


def main():
    data_dir = Path(__file__).parent.parent / 'data_csv'
    output_dir = Path(__file__).parent.parent / 'data'

    # Create output directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw cross-sections
    filepath = data_dir / '附件2_9个断面地形数据.csv'
    print("=" * 60)
    print("Module 1: Data Preprocessing")
    print("=" * 60)
    print(f"\nLoading: {filepath}")

    sections = load_cross_sections(str(filepath))
    dates_sorted = sorted(sections.keys())
    print(f"\nTotal: {len(sections)} cross-sections from {dates_sorted[0]} to {dates_sorted[-1]}")

    # Interpolate to uniform grid
    print("\n--- Interpolating to uniform 1m grid ---")
    interpolated = process_all_sections(sections, dx=1.0)

    # Save processed data
    output_path = output_dir / 'cross_sections_processed.csv'
    save_processed_data(interpolated, str(output_path))

    # Return for use by other modules
    return interpolated, sections


if __name__ == '__main__':
    main()
