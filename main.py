"""
NAME:        main.py
VERSION:     4.0 (Add multi batch export handling)
DESCRIPTION: Main script to run training and/or evaluation
             of ML models as defined in config.yaml
"""

# ==========================================================================
#                               IMPORTS                       
# ==========================================================================

    # Python Packages
import os
import yaml
import sys
import torch
from torch.utils.data import DataLoader

    # This is custom code found within Batt-ML/src/*
from src.data.dataset import BatteryDataset, TRAIN_CELLS, VAL_CELLS, TEST_CELLS
from src.models import get_model
from src.engine.trainer import train_model
from src.engine.evaluator import evaluate_model
from src.utils.logger import Logger
from src.utils.export import export_model

    # Import the configuration file 'config.yaml'
def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()


    # Determine if GPU should be enabled or not
device = torch.device("cuda" if torch.cuda.is_available() and 
                      CONFIG['device'] != 'cpu' else "cpu")



# ==========================================================================
#                            MAIN FUNCTION                         
# ==========================================================================

def main():

    # 1. Setup

    print(f"Running Experiment: {CONFIG['experiment_name']} on {device}")
    save_dir = os.path.join(CONFIG['output_dir'], CONFIG['experiment_name'])
    os.makedirs(save_dir, exist_ok = True)

    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok = True)


    # 2. Dataset
        # Get data location from config
    data_dir = CONFIG['data']['dir']

        # Get Training / Validation / Test Datasets
    train_cells = BatteryDataset(data_dir, TRAIN_CELLS, normalise=True)
    val_cells   = BatteryDataset(data_dir, VAL_CELLS, normalise=True)
    test_cells  = BatteryDataset(data_dir, TEST_CELLS, normalise=True)

        # Clump data into a format that can be passed straight into ML model
    train_loader = DataLoader(train_cells, batch_size=CONFIG['data']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_cells, batch_size=CONFIG['data']['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_cells, batch_size=1, shuffle=False)


    # 3. Build Model
    model = get_model(CONFIG).to(device)
    print(f"Initalised {CONFIG['model']} Model")


    # 4. Execution Mode
    mode = CONFIG['mode'] # Mode determines is code run train and/or evaluation



    # --- TRAINING SECTION --------------------------------------------------
    if "train" in mode:

        # Log any information printed to terminal
        original_stdout = sys.stdout
        log_filepath = os.path.join(log_dir, "train.txt")
        sys.stdout = Logger(log_filepath)

        # Run Training
        try:
            print(f"\n --- Training Experiment: {CONFIG['experiment_name']} ---")

            model_config = CONFIG['models'][CONFIG['model']]

            trained_model = train_model(
                model = model,
                train_loader = train_loader,
                val_loader = val_loader,
                device = device,
                epochs = model_config['epochs'],
                lr = model_config['learning_rate'],
                save_dir = save_dir,
                CONFIG = CONFIG
            )

            model = trained_model # Update model with best weights

            # Export model to ONNX filetype
            # Export different batch sizes as set in config file
            for bs in CONFIG['export']['batch_size']:

                # Create  specific batch size subfolder (e.g. test/bs1)
                bs_dir = os.path.join(save_dir, f"bs{bs}")
                os.makedirs(bs_dir, exist_ok=True)

                # Define exact file path (e.g. test/bs1/<model>.onnx)
                bs_export_path = os.path.join(bs_dir, f"{CONFIG['model']}.onnx")

                print(f"\nExporting {CONFIG['model']}" \
                        f"\n\tBatch size: {bs}" \
                        f"\n\tFile Path : {bs_export_path}")

                # Export the model with the specified batch size
                export_model(
                    model = model,
                    filepath = bs_export_path,
                    device = device,
                    batch_size=bs
                )


        finally:
            sys.stdout.close()
            sys.stdout = original_stdout



    # --- EXPORT SECTION (load checkpoint + re-export without retraining) ------
    if "export" in mode and "train" not in mode:

        checkpoint_path = os.path.join(save_dir, f"{CONFIG['model']}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: checkpoint not found at {checkpoint_path}")
            sys.exit(1)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")

        for bs in CONFIG['export']['batch_size']:

            bs_dir = os.path.join(save_dir, f"bs{bs}")
            os.makedirs(bs_dir, exist_ok=True)

            bs_export_path = os.path.join(bs_dir, f"{CONFIG['model']}.onnx")

            print(f"\nExporting {CONFIG['model']}" \
                    f"\n\tBatch size: {bs}" \
                    f"\n\tFile Path : {bs_export_path}")

            export_model(
                model = model,
                filepath = bs_export_path,
                device = device,
                batch_size=bs
            )


    # --- EVALUATION SECTION --------------------------------------------------
    if "eval" in mode:

        # Log any information printed to terminal
        original_stdout = sys.stdout
        log_filepath = os.path.join(log_dir, "eval.txt")
        sys.stdout = Logger(log_filepath)

        # Run evaluation
        try:
            print(f"\n --- Evaluation Experiment: {CONFIG['experiment_name']} ---")

            # Evaluate on Validation Set
            evaluate_model(
                model=model,
                device=device,
                data_dir=CONFIG['data']['dir'], 
                save_dir=save_dir, 
                cells_list=VAL_CELLS, 
                group_name="Validation Cells"
            )


            # Evaluate on Test Set
            evaluate_model(
                model = model,
                device = device,
                data_dir = CONFIG['data']['dir'],
                save_dir = save_dir,
                cells_list = TEST_CELLS,
                group_name = "Test Cells"
            )

        finally:
            sys.stdout.close()
            sys.stdout = original_stdout



# Executes the main function
if __name__ == "__main__":
    main()

