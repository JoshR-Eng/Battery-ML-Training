"""
NAME:           evaluate.py
VERSION:        1.2 (Subplot Graphs) 
DESCRIPTION:    Evaluates LSTM model with separated subplots for each cell
"""

# ==========================================================================
# --------                        IMPORTS                      --------
# ==========================================================================
import torch
import numpy as np
import os
import sys
import yaml
import matplotlib
matplotlib.use('Agg') # Runs matplotlib in headless mode
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from preprocessing.dataset import BatteryDataset, TRAIN_CELLS, VAL_CELLS, TEST_CELLS
from model.lstm import LSTMModel
from torch.utils.data import DataLoader


# ==========================================================================
# --------                  LOAD CONFIGURATION                  --------
# ==========================================================================

def load_config(config_name="config.yaml"):
    root_path = os.path.dirname(current_dir)
    config_path = os.path.join(root_path, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load the Config
CONFIG = load_config()

# LSTM Configuration
LSTM_CONF = CONFIG['LSTM']
lstm_param = LSTM_CONF['hyperparameters']
lstm_train = LSTM_CONF['training']

# Resolve Paths
PATH_CONFIG = CONFIG['Paths']
DATA_DIR = os.path.join(current_dir, PATH_CONFIG['data_dir'])
MODEL_PATH = os.path.join(current_dir, "../models/best_lstm.pth")

# Device Setup
if lstm_train['device'] == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(lstm_train['device'])


# ==========================================================================
# --------                        EVALUATION                      --------
# ==========================================================================

def evaluate():
    print(f"Evaluating Model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No model at {MODEL_PATH}")
        return

    # 1. Load Model Structure
    model = LSTMModel(
            input_size=lstm_param['input_size'],
            hidden_size=lstm_param['hidden_size'],
            num_layers=lstm_param['num_layers'],
            output_size=lstm_param['output_size'],
            dropout=lstm_param['dropout']
            ).to(device)

    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Define Test Sets
    test_sets = {
            "Validation": VAL_CELLS,
            "Test_Fixed": TEST_CELLS 
            }

    # 3. Iterate through Groups (Validation vs Test)
    for group_name, cell_list in test_sets.items():
        print(f"\n--- Processing Group: {group_name} ---")
        
        # Create a figure with N subplots (one per cell in the list)
        num_cells = len(cell_list)
        if num_cells == 0:
            print(f"   No cells found for {group_name}")
            continue

        # Adjust figure height based on number of cells (4 inches per cell)
        fig, axes = plt.subplots(num_cells, 1, figsize=(10, 4 * num_cells), constrained_layout=True)
        
        # If there's only 1 cell, matplotlib returns a single axis, not a list. Wrap it.
        if num_cells == 1: 
            axes = [axes]
        
        group_rmse_sum = 0.0
        valid_cells_count = 0

        # 4. Iterate through EACH CELL individually
        for i, cell_id in enumerate(cell_list):
            ax = axes[i]
            
            try:
                # Load ONLY this specific cell
                ds = BatteryDataset(DATA_DIR, [cell_id], normalise=True)
                
                if len(ds) == 0:
                    print(f"   Cell {cell_id}: No valid data found (Skipping)")
                    ax.text(0.5, 0.5, f"Cell {cell_id}: No Valid Cycles Found", 
                            ha='center', va='center', transform=ax.transAxes)
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
                
                # Denormalize
                preds = np.array(predictions) * 2.4 
                acts = np.array(actuals) * 2.4      
                
                # Metrics
                rmse = np.sqrt(np.mean((preds - acts)**2))
                group_rmse_sum += rmse
                valid_cells_count += 1
                
                print(f"   Cell {cell_id} | RMSE: {rmse:.4f} Ah")
                
                # Plotting this specific cell
                ax.plot(acts, label='Actual', color='black', linewidth=2)
                ax.plot(preds, label='Prediction', color='red',
                        linestyle='--', alpha=0.8)
             
                ax.set_title(f"Cell {cell_id} (RMSE: {rmse:.4f} Ah)")
                ax.set_ylabel("Capacity (Ah)")
                ax.set_xlabel("Cycle Number")
                ax.legend()
                ax.grid(True, alpha=0.3)

            except Exception as e:
                print(f"   Cell {cell_id} Error: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', 
                        va='center', transform=ax.transAxes)

        # 5. Save the Combined Figure for this Group
        save_filename = f"figures/results_lstm_{group_name}.png"
        plt.savefig(save_filename)
        plt.close() # clear memory
        
        print(f"   Plot saved to: {save_filename}")
        
        if valid_cells_count > 0:
            avg_rmse = group_rmse_sum / valid_cells_count
            print(f"   Average {group_name} RMSE: {avg_rmse:.4f} Ah")

if __name__ == "__main__":
    evaluate()
