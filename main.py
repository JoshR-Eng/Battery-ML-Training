"""
NAME:        main.py
VERSION:     1.0
DESCRIPTION: Main script to run training and/or evaluation
             of ML models as defined in config.yaml
"""

# ==========================================================================
# --------                      IMPORTS                         --------
# ==========================================================================

    # Python Packages
import os
import yaml
import torch
from torch.utils.data import DataLoader

    # This is custom code found within Battery-ML/src/*
from src.data.dataset import BatteryDataset, TRAIN_CELLS, VAL_CELLS, TEST_CELLS
from src.models import get_model
from src.engine.trainer import train_model
from src.engine.evaluator import evaluate_model

    # Import the configuration file 'config.yaml'
def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()


    # Determine if GPU should be enabled or not
device = torch.device("cuda" if torch.cuda.is_available() and 
                      CONFIG['device'] != 'cpu' else "cpu")



# ==========================================================================
# --------                  MAIN FUNCTION                         --------
# ==========================================================================

def main():

    # 1. Setup

    print(f"Running Experiment: {CONFIG['experiment_name']} on {device}")
    save_dir = os.path.join(CONFIG['output_dir'], CONFIG['experiment_name'])
    os.makedirs(save_dir, exist_ok = True)


    # 2. Dataset
        # Get data location from config
    data_dir = CONFIG['data']['dir']

        # Get Training / Validation / Test Datasets
    train_cells = BatteryDataset(data_dir, TRAIN_CELLS, normalise=True)
    val_cells   = BatteryDataset(data_dir, VAL_CELLS, normalise=True)
    test_cells  = BatteryDataset(data_dir, TEST_CELLS, normalise=True)

        # Clump data into a format that can be passed straight into ML model
    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)


    # 3. Build Model
    model = get_model(CONFIG).to(device)
    print(f"Initalised {CONFIG['model']} Model")


    # 4. Execution Mode
    mode = CONFIG['mode'] # Mode determines is code run train and/or evaluation

        # If train is in the mode, train the specified model
    if "train" in mode:
        print("\nStarting Training...")
        model_config = CONFIG['models'][cfg['model']]

        trained_model = train_model(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            device = device,
            epochs = model_config['epochs'],
            lr = model_config['learning_rate'],
            save_dir = save_dir
        )
        
        model = trained_model # Update model with best weights

        # If evaluation is in the mode, evaluate the model
    if "eval" in mode:
        print("\nStarting Evaluation...")
        evaluate_model(
            model = model,
            device = device,
            data_dir = CONFIG['data']['dir'],
            save_dir = save_dir,
            cells_list = TEST_CELLS,
            group_name = "Test Cells"
        )



# Executes the main function
if __name__ == "__main__":
    main()

