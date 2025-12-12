"""
NAME:           train.py
VERSION:        1.0
DESCRIPTION:    File to train ML algorithms on Q-V curve
"""

# ==========================================================================
# --------                        IMPORTS                      --------
# ==========================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import sys
import yaml

# Need to put root dir on path so I can asses ../data
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# From src/preprocessing/dataset.py
from preprocessing.dataset import BatteryDataset, TRAIN_CELLS, VAL_CELLS

# Gather defined ML models
from model.lstm import LSTMModel


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

# Resolve paths
PATH_CONFIG = CONFIG['Paths']
DATA_DIR = os.path.join(current_dir, PATH_CONFIG['data_dir'])
SAVE_DIR = os.path.join(current_dir, PATH_CONFIG['save_dir'])
os.makedirs(SAVE_DIR, exist_ok=True)

# Device Setup
if lstm_train['device'] == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(lstm_train['device'])

# ==========================================================================
# --------                  TRAINING ENGINE                    --------
# ==========================================================================

def train():
    print(f"Experiment: {CONFIG['experiment_name']}")

    # load all the data

        # Check data exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data Directory not at {DATA_DIR}")
        return

    print("     Loading Training Set...")
    try:
        train_ds = BatteryDataset(DATA_DIR, TRAIN_CELLS, normalise=True)
        val_ds = BatteryDataset(DATA_DIR, VAL_CELLS, normalise=True)
    except RuntimeError as e:
        print(f"ERROR (Dataset): {e}")
        return

    # Use config batch size
    train_loader = DataLoader(train_ds, batch_size=lstm_train['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=lstm_train['batch_size'],
                            shuffle=False)


    # Initialise Model
    print(f"Initialise Experiment")

    model = LSTMModel(
            input_size=lstm_param['input_size'],
            hidden_size=lstm_param['hidden_size'],
            num_layers=lstm_param['num_layers'],
            output_size=lstm_param['output_size'],
            dropout=lstm_param['dropout']
            ).to(device)


    # Setup Optimiser
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), 
                           lr=lstm_train['learning_rate'])

    # Training Loop
    print("\nStarting Training Loop...")
    start_time = time.time()
    best_val_rmse = float('inf')

    for epoch in range(lstm_train['epochs']):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Move to GPU/CPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward Pass
            optimiser.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward Pass
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)


        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)


                preds = model(X_val)
                batch_loss = criterion(preds, y_val)
                val_loss += batch_loss.item() * X_val.size(0)

        val_loss /= len(val_loader.dataset)
        val_rmse = np.sqrt(val_loss)

        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"\tEpoch {epoch+1:03d}/{lstm_train['epochs']} | " \
                    f"Train Loss: {train_loss:.6f} | " \
                    f"Val RMSE: {val_rmse:.5f}")
        
        # Checkpointing
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_path = os.path.join(SAVE_DIR, "best_lstm.pth")
            torch.save(model.state_dict(), save_path)

    total_time = time.time() - start_time
    print(f"\nDone!\n\n \tTotal Time: {total_time:.1f}s")
    print(f"\tBest RMSE: {best_val_rmse:.5f}")
    print(f"\tModel Saved: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    train()
