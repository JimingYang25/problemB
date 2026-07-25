"""
Module 3: Driving Factors Extraction
From 附件1 hourly water-sediment data, for each interval between
cross-section measurements, compute:
  - V_i: Cumulative runoff volume
  - M_i: Cumulative sediment transport
  - Q_peak,i: Maximum flood peak discharge
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_hydro_data(filepath: str) -> pd.DataFrame:
    """
    Load 附件1 hourly water-sediment data.
    Handles BOM encoding and missing values.
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Convert numeric columns, coerce errors
    df['水位(m)'] = pd.to_numeric(df['水位(m)'], errors='coerce')
    df['流量(m3/s)'] = pd.to_numeric(df['流量(m3/s)'], errors='coerce')
    df['含沙量(kg/m3)'] = pd.to_numeric(df['含沙量(kg/m3)'], errors='coerce')

    # Fill missing sediment concentration with interpolation
    df['含沙量(kg/m3)'] = df['含沙量(kg/m3)'].interpolate(method='linear')
    # Backfill/frontfill any remaining NaN
    df['含沙量(kg/m3)'] = df['含沙量(kg/m3)'].bfill().ffill()

    print(f"  Loaded {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  NaNs filled: 含沙量 remaining NaN = {df['含沙量(kg/m3)'].isna().sum()}")

    return df


def extract_driving_factors(
    hydro_df: pd.DataFrame,
    date_pairs: list,
    section_dates: list
) -> pd.DataFrame:
    """
    Extract V, M, Qpeak for each interval between cross-section measurements.

    Args:
        hydro_df: Hourly hydro-sediment data
        date_pairs: List of (start_date, end_date) tuples (strings)

    Returns:
        DataFrame with columns: 起始日期, 结束日期, V_累计径流量(m³),
                                M_累计输沙量(kg), Q_peak(m³/s)
    """
    results = []

    for start_date, end_date in date_pairs:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Select data in interval
        mask = (hydro_df['datetime'] >= start) & (hydro_df['datetime'] < end)
        interval = hydro_df[mask]

        if len(interval) == 0:
            print(f"  WARNING: No data for interval [{start_date}, {end_date})")
            results.append({
                '起始日期': start_date,
                '结束日期': end_date,
                'V_累计径流量(m³)': 0.0,
                'M_累计输沙量(kg)': 0.0,
                'Q_peak(m³/s)': 0.0,
                '记录数': 0
            })
            continue

        # Compute time deltas in seconds
        dt_seconds = interval['datetime'].diff().dt.total_seconds()
        dt_seconds.iloc[0] = 0  # First record has no previous

        # Cumulative runoff volume V = Σ Q(t) · Δt
        V = (interval['流量(m3/s)'].values * dt_seconds.values).sum()

        # Cumulative sediment transport M = Σ Q(t) · S(t) · Δt
        M = (interval['流量(m3/s)'].values *
             interval['含沙量(kg/m3)'].values *
             dt_seconds.values).sum()

        # Maximum flood peak discharge
        Q_peak = interval['流量(m3/s)'].max()

        results.append({
            '起始日期': start_date,
            '结束日期': end_date,
            'V_累计径流量(m³)': round(V, 0),
            'M_累计输沙量(kg)': round(M, 0),
            'Q_peak(m³/s)': round(Q_peak, 2),
            '记录数': len(interval)
        })

        print(f"  [{start_date} → {end_date}] "
              f"V={V:.2e} m³, M={M:.2e} kg, "
              f"Q_peak={Q_peak:.1f} m³/s ({len(interval)} records)")

    df = pd.DataFrame(results)
    return df


def main():
    from preprocess import load_cross_sections

    data_dir = Path(__file__).parent.parent / 'data_csv'
    output_dir = Path(__file__).parent.parent / 'data'

    print("=" * 60)
    print("Module 3: Driving Factors Extraction")
    print("=" * 60)

    # Load hydro data
    print("\n--- Loading 附件1 ---")
    hydro_file = data_dir / '附件1_逐小时水沙数据_2016-2021.csv'
    hydro_df = load_hydro_data(str(hydro_file))

    # Get cross-section dates
    print("\n--- Determining time intervals ---")
    sections_file = data_dir / '附件2_9个断面地形数据.csv'
    sections = load_cross_sections(str(sections_file))
    section_dates = sorted(sections.keys())

    print(f"  Cross-section dates: {section_dates}")

    # Build date pairs
    date_pairs = [(section_dates[i], section_dates[i + 1])
                  for i in range(len(section_dates) - 1)]

    # Extract driving factors
    print("\n--- Extracting Driving Factors ---")
    factors_df = extract_driving_factors(hydro_df, date_pairs, section_dates)

    # Save
    factors_df.to_csv(
        output_dir / 'driving_factors.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"\nSaved driving factors to {output_dir / 'driving_factors.csv'}")

    return factors_df, hydro_df


if __name__ == '__main__':
    main()
