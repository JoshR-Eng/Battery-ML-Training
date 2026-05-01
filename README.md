# Battery-ML-Training

A config-driven training pipeline for lithium-ion battery State-of-Health (SoH) prediction from Q-V discharge curves. Models are trained in PyTorch, evaluated on a stratified test split, and exported as multi-batch-size ONNX files ready for TensorRT compilation in [Edge-ML](https://github.com/JoshR-Eng/Edge-ML).

***

## Contents

- [Overview](#overview)
- [Pipeline Position](#pipeline-position)
- [Quick Start](#quick-start)
- [Configuration — `config.yaml`](#configuration--configyaml)
- [Training Modes](#training-modes)
- [Batch Training — `train_models.sh`](#batch-training--train_modelssh)
- [Training Engine](#training-engine)
- [Model Architectures](#model-architectures)
- [Adding a New Model](#adding-a-new-model)
- [Output Structure](#output-structure)
- [Dataset Splits](#dataset-splits)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

***

## Overview

The pipeline takes pre-processed Q-V curve tensors as input and trains sequence models to predict normalised battery capacity (SoH). All behaviour — from model selection and hyperparameters to export batch sizes — is controlled through a single `config.yaml` file without touching any Python source.

**Input:** Q-V discharge curve tensors (`data/tensors_qv/`) — 120-point voltage sequences per cell per cycle  
**Output:** `.pth` checkpoints + multi-batch-size `.onnx` exports + evaluation logs  
**Target:** Normalised capacity in , denormalised against a 2.4 Ah nominal capacity

***

## Pipeline Position

```
Battery-ML-Training  ──(ONNX exports)──►  Edge-ML
   Data preparation                         TensorRT compilation
   Model training                           FP32 / FP16 / INT8 engines
   ONNX export                              Benchmarking on Jetson Orin Nano
```

ONNX files are exported at multiple batch sizes (default: `[1, 32, 96]`) so that `Edge-ML` can compile and benchmark each configuration independently.

***

## Quick Start

```bash
# Clone and install
git clone https://github.com/JoshR-Eng/Battery-ML-Training.git
cd Battery-ML-Training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the model set in config.yaml
python main.py

# Train all five architectures sequentially
./train_models.sh

# Train a specific subset with a named experiment
./train_models.sh --name v4 LSTM GRU TCN
```

***

## Configuration — `config.yaml`

All pipeline behaviour is controlled here. No Python edits are needed for routine experiments.

```yaml
experiment_name: "01/GRU"       # Output folder: results/01/GRU/
output_dir: "./results"
device: "auto"                   # cuda / cpu / auto

mode: "train-eval"               # train | eval | train-eval | export

model: "GRU"                     # Active model (must match a key in 'models:')

data:
  dir: "./data/tensors_qv"
  batch_size: 32
  num_workers: 4
  nominal_capacity: 2.4          # Ah — used to denormalise outputs for evaluation

export:
  batch_size: [1, 32, 96]        # ONNX exports generated for each batch size

models:
  GRU:
    input_size: 1
    hidden_size: 64
    num_layers: 2
    dropout: 0.1
    output_size: 1
    learning_rate: 0.001
    epochs: 150
  # ... other model configs below
```

### Key config parameters

| Parameter | Options | Effect |
|-----------|---------|--------|
| `mode` | `train`, `eval`, `train-eval`, `export` | Controls which pipeline stages execute |
| `model` | `LSTM`, `GRU`, `CNN-LSTM`, `CNN-GRU`, `TCN` | Selects architecture; config reads matching `models:` block |
| `device` | `auto`, `cuda`, `cpu` | `auto` selects GPU if available |
| `export.batch_size` | list of ints | One `.onnx` file is written per batch size under `results/.../bs<N>/` |
| `data.batch_size` | int | DataLoader batch size during training |

Each entry under `models:` is an independent hyperparameter block. Changing `model:` at the top of the file is all that is needed to switch architecture — the pipeline reads the matching block automatically.

***

## Training Modes

Set `mode:` in `config.yaml` to control what runs:

| Mode | Behaviour |
|------|-----------|
| `train` | Train from scratch, save best checkpoint, export ONNX |
| `eval` | Load existing checkpoint, evaluate on val + test sets |
| `train-eval` | Train then immediately evaluate (default) |
| `export` | Re-export ONNX from existing `.pth` without retraining |

The `export` mode is useful when changing `export.batch_size` without wanting to retrain.

***

## Batch Training — `train_models.sh`

Trains multiple models sequentially, auto-patching `config.yaml` between runs and restoring it on completion.

```bash
./train_models.sh                             # All 5 architectures
./train_models.sh LSTM GRU TCN               # Specific subset
./train_models.sh --name v4 LSTM GRU CNN-LSTM CNN-GRU TCN   # Named experiment
```

After all runs complete, the script prints a results table and highlights the best model by test RMSE:

```
Model           | Val RMSE     | Test RMSE    | Time
------------------------------------------------------------------------
LSTM            | 0.03412      | 0.03508      | 312s
GRU             | 0.03187      | 0.03291      | 287s
...
BEST MODEL: GRU
   Test RMSE: 0.03291 Ah
   Location: ./results/v4/GRU/
```

> `config.yaml` is backed up as `config.yaml.backup` before any modifications and restored after.

***

## Training Engine

`src/engine/trainer.py` implements the core training loop with three built-in features that require no configuration changes:

- **Adam optimiser** with per-model learning rate from `config.yaml`
- **ReduceLROnPlateau scheduler** — halves learning rate after 10 epochs of no validation improvement (`factor=0.5, patience=10`)
- **Early stopping** — halts training after 20 consecutive epochs with no improvement in validation RMSE, then restores the best checkpoint before returning

Only the best model weights (lowest validation RMSE) are saved as `<model>.pth`. The ONNX export uses these best weights.

***

## Model Architectures

Five sequence-to-scalar architectures are implemented, all consuming a 120-point voltage sequence and producing a single SoH scalar.

| Model | File | Architecture | Notes |
|-------|------|-------------|-------|
| `LSTM` | `lstm.py` | 2-layer LSTM → FC | Recurrent baseline |
| `GRU` | `gru.py` | 2-layer GRU → FC | Fewer parameters than LSTM |
| `CNN-LSTM` | `cnn_lstm.py` | 3-stage CNN → 2-layer LSTM → FC | CNN extracts local voltage features first |
| `CNN-GRU` | `cnn_gru.py` | 3-stage CNN → 2-layer GRU → FC | Same as CNN-LSTM but lighter recurrent stage |
| `TCN` | `tcn.py` | Dilated causal convolutions → FC | Parallelisable; fastest at inference |

CNN stages use progressive channels `[16, 32, 64]` with `kernel_size=5` (configurable per model in `config.yaml`) to capture patterns across 5 consecutive voltage points.

***

## Adding a New Model

The model registry in `src/models/__init__.py` uses a `get_model(config)` dispatcher. Adding a new architecture requires four steps:

**1.** Create `src/models/my_model.py` implementing a `nn.Module` with a standard forward signature:

```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ...):
        super().__init__()
        # define layers

    def forward(self, x):           # x: (batch, seq_len, input_size)
        ...
        return output               # (batch, 1)
```

**2.** Import and register it in `src/models/__init__.py`:

```python
from .my_model import MyModel

# Inside get_model():
elif model_name == "MyModel":
    return MyModel(
        input_size  = model_params['input_size'],
        hidden_size = model_params['hidden_size'],
        output_size = model_params['output_size'],
    )
```

**3.** Add a config block to `config.yaml` under `models:`:

```yaml
models:
  MyModel:
    input_size: 1
    hidden_size: 64
    output_size: 1
    learning_rate: 0.001
    epochs: 150
```

**4.** Set `model: "MyModel"` at the top of `config.yaml` and run `python main.py`.

No other files need to change. The training, evaluation, and ONNX export paths all read the model name from config.

***

## Output Structure

```
results/
└── <experiment_name>/          # e.g. v4/GRU/
    ├── GRU.pth                 # Best checkpoint (lowest val RMSE)
    ├── bs1/
    │   └── GRU.onnx            # ONNX export at batch size 1
    ├── bs32/
    │   └── GRU.onnx
    ├── bs96/
    │   └── GRU.onnx
    └── logs/
        ├── train.txt           # Full training log (epoch losses, LR changes)
        └── eval.txt            # Evaluation results — val + test RMSE/MAE
```

The ONNX exports in `bs<N>/` directories map directly to the `models/<folder>/` layout expected by `Edge-ML`.

***

## Dataset Splits

The 80-cell dataset is split into three non-overlapping groups. The same split is mirrored in `Edge-ML/configs.yaml` to ensure INT8 calibration data never overlaps with the test set.

| Split | Proportion | Cells | Purpose |
|-------|-----------|-------|---------|
| Train | 70% | 47 cells | Model training + INT8 calibration (in Edge-ML) |
| Val   | 15% | 12 cells | Scheduler / early stopping / hyperparameter tuning |
| Test  | 15% | 13 cells | Held-out evaluation only — never used for training or calibration |

Splits are defined as hardcoded cell ID lists in `src/data/dataset.py` (`TRAIN_CELLS`, `VAL_CELLS`, `TEST_CELLS`).

***

## Project Structure

```
Battery-ML-Training/
├── main.py                      # Entry point — reads config, runs pipeline
├── config.yaml                  # All hyperparameters and pipeline settings
├── train_models.sh              # Batch training with results summary
├── requirements.txt
├── data/
│   └── tensors_qv/              # Pre-processed Q-V tensors (input data)
├── results/                     # Generated at runtime
└── src/
    ├── data/
    │   └── dataset.py           # BatteryDataset, cell split definitions
    ├── models/
    │   ├── __init__.py          # get_model() dispatcher — model registry
    │   ├── lstm.py
    │   ├── gru.py
    │   ├── cnn_lstm.py
    │   ├── cnn_gru.py
    │   └── tcn.py
    ├── engine/
    │   ├── trainer.py           # Training loop, early stopping, checkpointing
    │   └── evaluator.py        # Val/test evaluation, metric logging
    └── utils/
        ├── export.py            # ONNX export (multi-batch-size)
        └── logger.py            # stdout → file tee logger
```

***

## Requirements

```
torch>=2.0.0
numpy
pandas
matplotlib
scipy
onnx
onnxruntime
pyyaml
```

```bash
pip install -r requirements.txt
```
