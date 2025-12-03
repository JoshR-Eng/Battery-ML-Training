import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import torch

# ==========================================================================
# --------                  CONFIGURATION                      --------
# ==========================================================================

INPUT_DIR = "data/data_processed"
OUTPUT_DIR = "data/tensors_qv"
VOLTAGE_POINTS = 120
V_MIN = 3.0
V_MAX = 4.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================================
# --------                  QV CURVE FUNC.                     --------
# ==========================================================================

def extract_qv_curve(cycle_df, points=VOLTAGE_POINTS):
    """
    Interpolate Capacity (Q) over a fixed Voltage (V) grid
    """
    # GET DATA
    v = cycle_df['voltage(v)'].values
    t = cycle_df['time_s'].values
    i = cycle_df['current(a)'].values

    # Capacitance (Q) Integration: Q(t) = sum(I * dt)
    # prepend=t[0] ensures the first delta is 0, avoiding massive jumps
    dt = np.diff(t, prepend=t[0])
    q = np.cumsum(np.abs(i * dt)) / 3600.0

    # SORT BY VOLTAGE FOR INTERPOLATION
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    q_sorted = q[sort_idx]

    # Remove duplicates
    # FIX: Typo 'return_idex' -> 'return_index'
    v_unique, unique_idx = np.unique(v_sorted, return_index=True)
    q_unique = q_sorted[unique_idx]

    # CREATE FIXED VOLTAGE INTERVAL GRID
    v_grid = np.linspace(V_MIN, V_MAX, VOLTAGE_POINTS)

    # INTERPOLATE Q MAPPED TO V-GRID
    f_q = interp1d(v_unique, q_unique, kind='linear',
                   bounds_error=False, fill_value="extrapolate")
    q_interpolated = f_q(v_grid)

    return q_interpolated

# ==========================================================================
# --------                  MAIN PROCESSING                    --------
# ==========================================================================

def process_datasets():
    # Only pick up .parquet files to avoid confusion with folders
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')])

    if not files:
        print(f"No parquet files found in {INPUT_DIR}")
        return

    for filename in files:
        print(f"Processing {filename}...")
        
        try:
            df = pd.read_parquet(os.path.join(INPUT_DIR, filename))
        except Exception as e:
            print(f"   Error reading file: {e}")
            continue

        cycles_inputs = []
        cycles_targets = []

        grouped = df.groupby('cycle_index')

        for cycle_idx, group in grouped:
            # Filter for Discharge phase
            discharge = group[group['current(a)'] < -0.1 ]
            
            # Skips short cycles assuming garbage data
            if len(discharge) < 50: continue

            # Extract target (Total Max Cap. of this cycle)
            t_vals = discharge['time_s'].values
            
            # FIX: Physics bug. Must prepend first timestamp, not 0.
            dt = np.diff(t_vals, prepend=t_vals[0])
            total_capacity = np.abs(np.sum(discharge['current(a)'].values * dt )) / 3600.0

            # Extract Input Features (120-point QV Curve)
            # We removed the generic try/except so you can see if something breaks
            try:
                qv_curve = extract_qv_curve(discharge, points=VOLTAGE_POINTS)
                cycles_inputs.append(qv_curve)
                cycles_targets.append(total_capacity)
            except Exception as e:
                print(f"Skipped Cycle {cycle_idx} due to error: {e}")
                pass

        if len(cycles_inputs) > 0:
            x = np.array(cycles_inputs, dtype=np.float32)
            y = np.array(cycles_targets, dtype=np.float32)

            # Save data
            save_name = filename.replace('.parquet', '.pt')
            torch.save({'X': torch.tensor(x),
                        'y': torch.tensor(y)},
                        os.path.join(OUTPUT_DIR, save_name))
            
            print(f"Saved {len(cycles_inputs)} cycles as {save_name}")
        else:
            print(f"No valid cycles found in {filename}")

if __name__ == "__main__":
    process_datasets()
