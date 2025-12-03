"""
NAME:           evaluate.py
VERSION:        1.0
DESCRIPTION:    Evaluates lstm model with new data
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
# --------                  LOAD CONFIGURATION                 --------
# ==========================================================================

def load_config(config_name="config.yaml"):
    # config.yaml must be `./../` relative to this file
    root_path = os.path.dirname(current_dir)
    config_path = os.path.join(root_path, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load the Config
CONFIG = load_config()

# LSTM Confguration
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
# --------                       EVALUATION                      --------
# ==========================================================================


def evaluate():
    print(f"Evaluating Model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No model at {MODEL_PATH}")
        return


    # Load Model Structure
    model = LSTMModel(
            input_size=lstm_param['input_size'],
            hidden_size=lstm_param['hidden_size'],
            num_layers=lstm_param['num_layers'],
            output_size=lstm_param['output_size'],
            dropout=lstm_param['dropout']
            ).to(device)

    # Load the trained weights
    model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device))
    model.eval()


    # Define Test Sets
    test_sets = {
            "Validation": VAL_CELLS,
            "Test_Fixed": TEST_CELLS
            }

    results = {}

    for name, cells in test_sets.items():
        print(f"\n--- Testing on {name} ---")
        try:
            ds = BatteryDataset(DATA_DIR, cells, normalise=True)
            # Batch size 1 to evaluate cycle-by-cycle
            loader = DataLoader(ds, batch_size=1, shuffle=False)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
            
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                predictions.append(pred.item())
                actuals.append(y.item())
        
        # Calculate Metrics (Denormalise by multiplying by 2.4Ah)
        preds = np.array(predictions) * 2.4 
        acts = np.array(actuals) * 2.4      
        
        rmse = np.sqrt(np.mean((preds - acts)**2))
        mae = np.mean(np.abs(preds - acts))
        
        results[name] = rmse
        print(f"\tRMSE: {rmse:.4f} Ah")
        print(f"\tMAE:  {mae:.4f} Ah")
        
        # --- PLOTTING ---
        plt.figure(figsize=(12, 6))
        
        # Plot only the first 150 cycles to keep it readable
        limit = min(150, len(acts))
        
        plt.plot(acts[:limit], label='Actual Capacity',
                 color='black', linewidth=2)
        plt.plot(preds[:limit], label='LSTM Prediction',
                 color='red', linestyle='--', alpha=0.8)
        
        plt.title(f"LSTM Performance on {name} Data")
        plt.ylabel("Capacity (Ah)")
        plt.xlabel("Cycle Number")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # SAVE instead of show
        save_filename = f"results_lstm_{name}.png"
        plt.savefig(save_filename)
        plt.close() # Close memory to prevent overlapping plots
        
        print(f"Plot saved to: {save_filename}")

if __name__ == "__main__":
    evaluate()
