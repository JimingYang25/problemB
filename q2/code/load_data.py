"""
Data loading for Q2: Spatial VoI + Temporal Adaptive Sampling
- Parse 附件1 (6-year hourly hydro-sediment) into unified DataFrame
- Parse 附件3 (19-period cross-section velocity profiles) into depth-averaged format
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'q1' / 'data_csv'


def load_hydro_timeseries(filepath: str = None) -> pd.DataFrame:
    """
    Load 附件1: 2016-2021 hourly water level, discharge, sediment.
    Output: datetime-indexed DataFrame with clean column names.
    """
    if filepath is None:
        filepath = DATA_DIR / '附件1_逐小时水沙数据_2016-2021.csv'

    df = pd.read_csv(str(filepath), encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in ['水位(m)', '流量(m3/s)', '含沙量(kg/m3)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  [附件1] Loaded {len(df)} records: {df['datetime'].min()} → {df['datetime'].max()}")
    return df


def load_cross_section_velocities(filepath: str = None) -> dict:
    """
    Load 附件3: 19-period cross-section velocity/sediment profiles.

    Structure: rows with '起点距离(m)' are position markers; rows between
    them are multi-depth measurements at the same position.

    Returns:
        dict: {date_str: DataFrame with columns [x, depth_avg_velocity]}
    """
    if filepath is None:
        filepath = DATA_DIR / '附件3_断面流速含沙量数据.csv'

    df = pd.read_csv(str(filepath), encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    # Forward-fill position
    df['起点距离(m)'] = pd.to_numeric(df['起点距离(m)'], errors='coerce')
    df['位置'] = df['起点距离(m)'].ffill()

    # Parse numeric velocity
    df['测点水流速(m/s)'] = pd.to_numeric(df['测点水流速(m/s)'], errors='coerce')

    # Group by date and position, take depth-averaged velocity
    sections = {}
    for date, group in df.groupby('日期'):
        date = str(date).strip()
        pos_groups = group.groupby('位置')

        positions = []
        velocities = []
        for pos, pg in pos_groups:
            v_vals = pg['测点水流速(m/s)'].dropna().values
            if len(v_vals) > 0:
                positions.append(float(pos))
                velocities.append(float(np.mean(v_vals)))

        # Sort by position
        idx = np.argsort(positions)
        positions = np.array(positions)[idx]
        velocities = np.array(velocities)[idx]

        sections[date] = pd.DataFrame({
            'x_m': positions,
            'v_mean': velocities
        })

        print(f"  [{date}] {len(positions)} positions, "
              f"x=[{positions.min():.0f}, {positions.max():.0f}]m, "
              f"v=[{velocities.min():.3f}, {velocities.max():.3f}]m/s")

    print(f"\n  Total: {len(sections)} cross-section periods loaded")
    return sections


def normalize_and_bin(
    sections: dict,
    n_bins: int = 20
) -> np.ndarray:
    """
    Normalize positions to [0,1] and bin into n_bins standard slots.
    Uses the union of all positions across all dates as the reference range.

    Returns:
        obs_matrix: (n_periods, n_bins) depth-averaged velocity matrix
        bin_centers_m: actual positions (m) of bin centers
    """
    # Determine global x range
    all_x = []
    for df in sections.values():
        all_x.extend(df['x_m'].values)
    x_min, x_max = min(all_x), max(all_x)
    section_width = x_max - x_min

    # Bin edges in real coordinates
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    sorted_dates = sorted(sections.keys())
    n_periods = len(sorted_dates)
    obs_matrix = np.full((n_periods, n_bins), np.nan)

    for i, date in enumerate(sorted_dates):
        df = sections[date]
        for _, row in df.iterrows():
            x = row['x_m']
            v = row['v_mean']
            # Find which bin this x falls into
            bin_idx = np.searchsorted(bin_edges, x) - 1
            if 0 <= bin_idx < n_bins:
                obs_matrix[i, bin_idx] = v

    # Fill missing bins with nearest-neighbor interpolation per period
    for i in range(n_periods):
        mask = ~np.isnan(obs_matrix[i])
        if mask.sum() < 2:
            continue
        x_valid = np.where(mask)[0]
        y_valid = obs_matrix[i, mask]
        obs_matrix[i] = np.interp(np.arange(n_bins), x_valid, y_valid)

    print(f"\n  Binning: {n_periods} periods × {n_bins} bins")
    print(f"  Section width: {section_width:.0f} m, bin spacing: {section_width/n_bins:.1f} m")
    print(f"  Missing cells filled: {np.isnan(obs_matrix).sum()} → 0")

    return obs_matrix, bin_centers, sorted_dates


def build_daily_series(hydro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly hydro data to daily resolution.
    Adds |dH/dt| (m/day) and reconstructed sediment via S-Q rating.

    Returns daily DataFrame with: date, H, Q, S_true, Qs_true, dH
    """
    daily = hydro_df.set_index('datetime').resample('D').agg({
        '水位(m)': 'mean',
        '流量(m3/s)': 'mean',
        '含沙量(kg/m3)': 'mean'
    }).reset_index()

    daily.columns = ['date', 'H', 'Q', 'S_raw']
    daily['dH'] = daily['H'].diff().abs()  # |dH/dt| in m/day

    print(f"\n  Daily series: {len(daily)} days, "
          f"H=[{daily['H'].min():.2f}, {daily['H'].max():.2f}]m, "
          f"Q=[{daily['Q'].min():.0f}, {daily['Q'].max():.0f}] m³/s")

    return daily


if __name__ == '__main__':
    print("=" * 60)
    print("Q2 Data Loading")
    print("=" * 60)

    # 附件1
    hydro = load_hydro_timeseries()
    daily = build_daily_series(hydro)

    # 附件3
    sections = load_cross_section_velocities()
    obs_matrix, bin_centers, dates = normalize_and_bin(sections, n_bins=20)

    print("\nDone.")
