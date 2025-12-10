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

# STRICT LIMITS (The Microsoft Fix)
MAX_PHYSICAL_CAPACITY = 3.0  # Ah (Nominal is 2.4, so 3.0 is definitely error)
MIN_CYCLES_REQUIRED = 10     # If a file has fewer than 10 good cycles, skip it

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================================
# --------                  QV CURVE FUNC.                    --------
# ==========================================================================

def extract_qv_curve(cycle_df, points=VOLTAGE_POINTS):
    v = cycle_df['voltage(v)'].values
    t = cycle_df['time_s'].values
    i = cycle_df['current(a)'].values

    # ROBUST INTEGRATION
    # 1. Reset time to 0 for this cycle (Crucial for dt accuracy)
    t = t - t[0]
    
    # 2. Calculate dt
    dt = np.diff(t, prepend=0)
    
    # 3. Integrate Capacity: Q = Sum(I * dt)
    q = np.cumsum(np.abs(i * dt)) / 3600.0

    # 4. Interpolation
    # Sort by voltage to ensure monotonically increasing x-axis for interp1d
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    q_sorted = q[sort_idx]

    # Remove duplicates (Voltage must be unique for interpolation)
    v_unique, unique_idx = np.unique(v_sorted, return_index=True)
    q_unique = q_sorted[unique_idx]

    # Map to fixed grid
    v_grid = np.linspace(V_MIN, V_MAX, VOLTAGE_POINTS)
    f_q = interp1d(v_unique, q_unique, kind='linear',
                   bounds_error=False, fill_value="extrapolate")
    q_interpolated = f_q(v_grid)

    # 5. VARIANCE FEATURE (Tier D Model Prep)
    # This matches the "Severson" variance feature for baseline comparison
    q_variance = np.var(q_interpolated)

    return q_interpolated, q_variance

# ==========================================================================
# --------                  MAIN PROCESSING                    --------
# ==========================================================================

def process_datasets():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')])
    if not files:
        print(f"No parquet files found in {INPUT_DIR}")
        return

    print(f"Processing {len(files)} files...")

    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(INPUT_DIR, filename))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        cycles_inputs = []   # The Q-V Curve (Vector)
        cycles_targets = []  # The Capacity (Scalar)
        cycles_vars = []     # The Variance (Scalar - for Severson Baseline)

        grouped = df.groupby('cycle_index')

        for cycle_idx, group in grouped:
            # 1. Strict Discharge Filter
            # Current must be significantly negative to avoid "Rest" noise
            discharge = group[group['current(a)'] < -0.05] 

            if len(discharge) < 50: 
                continue

            # 2. Calculate Target Capacity (Ground Truth)
            t_vals = discharge['time_s'].values
            dt = np.diff(t_vals, prepend=t_vals[0])
            total_capacity = np.abs(
                    np.sum(discharge['current(a)'].values * dt )
                    ) / 3600.0

            # --- THE "MICROSOFT FIX" (Sanity Check) ---
            if total_capacity > MAX_PHYSICAL_CAPACITY or total_capacity < 0.1:
                # Silently skip the "Ghost" spikes (160Ah)
                continue
            # ------------------------------------------

            # 3. Extract Input Features
            try:
                qv_curve, q_var = extract_qv_curve(discharge, 
                                                   points=VOLTAGE_POINTS)
                
                cycles_inputs.append(qv_curve)
                cycles_targets.append(total_capacity)
                cycles_vars.append(q_var)
                
            except Exception as e:
                pass

        # Save only if enough data
        if len(cycles_inputs) > MIN_CYCLES_REQUIRED:
            x = np.array(cycles_inputs, dtype=np.float32)
            y = np.array(cycles_targets, dtype=np.float32)
            v = np.array(cycles_vars, dtype=np.float32)

            save_name = filename.replace('.parquet', '.pt')
            torch.save({
                'X': torch.tensor(x),      # Q-V Curve (For TCN/LSTM)
                'y': torch.tensor(y),      # Target Capacity
                'var': torch.tensor(v)     # Variance (For Linear Baseline)
            }, os.path.join(OUTPUT_DIR, save_name))
            
            print(f"\tSaved {save_name} ({len(cycles_inputs)} cycles)")
        else:
            print(f"\tSkipping {filename} (Not enough valid cycles)")

if __name__ == "__main__":
    process_datasets()
