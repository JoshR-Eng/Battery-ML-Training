# Battery Capacity Prediction using Deep Learning

Machine learning models for predicting lithium-ion battery capacity degradation from discharge voltage curves.
This repository is also tied to [Edge Deployment](https://github.com/JoshR-Eng/Edge-ML) repository in which all these models are quantised and evaluated on a Jetson Orin Nano 4GB.

## Features

- **Multiple Architectures**: LSTM, GRU, CNN-LSTM, CNN-GRU, TCN
- **Physics-Based Features**: Q-V curve extraction from discharge cycles
- **Stratified Dataset Split**: Fair evaluation across all charging protocols
- **Automated Training Pipeline**: Batch training script for model comparison

## Quick Start

### Standard Training
```bash
# Train single model
python main.py

# Train all models
./train_models.sh
```


## Model Comparison

| Model | Parameters | RMSE (Ah) | Inference Speed |
|-------|------------|-----------|-----------------|
| LSTM | ~50K | TBD | Baseline |
| GRU | ~37K | TBD | 0.7× LSTM |
| CNN-LSTM | ~85K | TBD | 1.4× LSTM |
| CNN-GRU | ~68K | TBD | 1.2× LSTM |
| TCN | ~42K | TBD | 0.9× LSTM |


## Project Structure

```
batt_ml/
├── main.py                      # Standard training
├── config.yaml                  # Configuration
├── train_models.sh              # Batch training
├── src/
│   ├── data/dataset.py         # Dataset class
│   ├── models/                 # Model architectures
│   │   ├── lstm.py
│   │   ├── cnn_lstm.py
│   │   └── ...
│   ├── engine/
│   │   ├── trainer.py         # Standard training
│   └── utils/
│       ├── qv_curve.py        # Q-V curve preprocessing
│       └── export.py          # ONNX export
└── results/                    # Model checkpoints & plots
```

## Requirements

```
torch>=2.0.0
numpy
pandas
matplotlib
scipy
onnx
onnxruntime
```

Install: `pip install -r requirements.txt`
