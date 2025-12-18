"""
NAME:           evaluate.py
VERSION:        1.3 (Make func. to call from main.py) 
DESCRIPTION:    Evaluates LSTM model with separated subplots for each cell
"""

# ==========================================================================
# --------                        IMPORTS                      --------
# ==========================================================================
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Runs matplotlib in headless mode
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from src.data.dataset import BatteryDataset


# ==========================================================================
# --------                        EVALUATION                      --------
# ==========================================================================

def evaluate_model(model, device, data_dir, save_dir, cells_list,
                   group_name = "Test"):

    print(f"Evaluating Cell Data: {group_name}")

    model.eval()


    # 1. Setup Plotting Figure
    num_cells = len(cells_list)
    if num_cells == 0:
        print(f"   No cells provided for {group_name}")
        return

        # Dynamic height: 4 inches per cell
    fig, axes = plt.subplots(num_cells, 1, figsize=(10, 4 * num_cells), constrained_layout=True)
    if num_cells == 1: 
        axes = [axes] # Ensure iterable

    group_rmse_sum = 0.0
    valid_cells_count = 0


    # 2. Iterate Per Cell (Crucial for visualization)
    for i, cell_id in enumerate(cells_list):
        ax = axes[i]
        
        try:
            # Load ONLY this specific cell
            # We construct a temporary dataset just for this loop
            ds = BatteryDataset(data_dir, [cell_id], normalise=True)
            
            if len(ds) == 0:
                ax.text(0.5, 0.5, f"Cell {cell_id}: No Data", ha='center', va='center')
                continue

            loader = DataLoader(ds, batch_size=1, shuffle=False)
            
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for X, y in loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    
                    predictions.append(pred.item())
                    actuals.append(y.item())
            
    # 3. Denormalize (Nominal Capacity = 2.4Ah)
            #    If you change nominal capacity, pass it as an arg!
            preds = np.array(predictions) * 2.4 
            acts = np.array(actuals) * 2.4      
    

    # 4. Metrics
            rmse = np.sqrt(np.mean((preds - acts)**2))
            group_rmse_sum += rmse
            valid_cells_count += 1
            
            print(f"   Cell {cell_id} | RMSE: {rmse:.4f} Ah")
            
            # 5. Plot
            ax.plot(acts, label='Actual', color='black', linewidth=2)
            ax.plot(preds, label='Prediction', color='red', linestyle='--', alpha=0.8)
            ax.set_title(f"Cell {cell_id} (RMSE: {rmse:.4f} Ah)")
            ax.set_ylabel("Capacity (Ah)")
            ax.set_xlabel("Cycle Number")
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"   Cell {cell_id} Error: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')


    # 6. Save Combined Figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"results_{group_name}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"   Plot saved to: {save_path}")
    
    if valid_cells_count > 0:
        avg_rmse = group_rmse_sum / valid_cells_count
        print(f"   Average {group_name} RMSE: {avg_rmse:.4f} Ah")
        return avg_rmse
    return 0.0
