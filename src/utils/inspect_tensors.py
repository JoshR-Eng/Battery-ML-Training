import torch
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# CONFIG
TENSOR_DIR = "data/tensors_qv"
VOLTAGE_POINTS = 120
V_MIN = 3.0
V_MAX = 4.2

def inspect_random_file():
    # 1. Pick a random file
    files = [f for f in os.listdir(TENSOR_DIR) if f.endswith('.pt')]
    if not files:
        print("No .pt files found!")
        return

    filename = random.choice(files)
    filepath = os.path.join(TENSOR_DIR, filename)
    print(f"Inspecting: {filename}")

    # 2. Load the Tensor
    data = torch.load(filepath)
    X = data['X']  # Shape: (Num_Cycles, 120)
    y = data['y']  # Shape: (Num_Cycles,)

    print(f"   Shape of Inputs (X): {X.shape}")
    print(f"   Shape of Targets (y): {y.shape}")
    print(f"   Example Target (Capacity): {y[0].item():.4f} Ah")

    # 3. Visualization
    # We need to recreate the Voltage Grid (X-axis) for plotting
    v_grid = np.linspace(V_MIN, V_MAX, VOLTAGE_POINTS)

    plt.figure(figsize=(10, 6))
    
    # Plot 5 random cycles from this battery
    num_cycles = X.shape[0]
    indices = random.sample(range(num_cycles), min(5, num_cycles))
    
    for idx in indices:
        # X[idx] is the Q-curve (Capacity vs Voltage)
        plt.plot(v_grid, X[idx].numpy(), label=f'Cycle {idx} (Cap: {y[idx]:.2f} Ah)')

    plt.title(f"Q-V Curves for {filename}")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Accumulated Capacity (Q)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    inspect_random_file()
