# Evidence Record

Baselines, architectures, tuned configurations, training protocol and results for [README.md](README.md).

## Baselines and ablations

### Sanity baselines

Zero and median predictors (`runs/zero`, `runs/median`) give a floor that the trained models should beat.

| Model | Purpose | Status |
|---|---|---|
| CNN | Convolutional baseline | Done |
| LSTM | Recurrent baseline | Done |
| BERT4NILM | NILM-specific Transformer baseline | Planned |
| MOMENT-small (random init) | To measure architectural gains | Planned |
| MOMENT-small (frozen) | To measure frozen feature gains | Planned |
| MOMENT-small (full fine-tuning) | To measure NILM adaptation gains | Planned |

## Architectures

### CNN (Zhang et al., [arXiv:1612.09106](https://arxiv.org/abs/1612.09106))

[src/models/cnn.py](src/models/cnn.py)

```
Input [B, 512]
→ Conv1d(1→30, k=10) → ReLU
→ Conv1d(30→30, k=8) → ReLU
→ Conv1d(30→40, k=6) → ReLU
→ Conv1d(40→50, k=5) → ReLU
→ Conv1d(50→50, k=5) → ReLU
→ Flatten → Linear(→1024) → ReLU → Dropout → Linear(1024→1)
```

### LSTM (adapted from Kelly & Knottenbelt, [arXiv:1507.06594](https://arxiv.org/abs/1507.06594))

[src/models/lstm.py](src/models/lstm.py)

```
Input [B, 512]
→ Conv1d(1→16, k=4, length-preserving padding)
→ BiLSTM(hidden h1) → BiLSTM(hidden h2)
→ hidden state at the window midpoint
→ Linear(2·h2→128) → Tanh → Linear(128→1)
```

Default hidden sizes are (h1, h2) = (128, 256).

## Tuned configurations

### CNN

| Parameter | Value |
|---|---|
| Conv channels | 30, 30, 40, 50, 50 |
| Kernel sizes | 10, 8, 6, 5, 5 |
| Strides | 1, 1, 1, 1, 1 |
| Dense units | 1024 |
| Dropout | 0.0 |
| Batch size | 512 |
| Learning rate | 3e-4 |
| Weight decay | 0.0 |
| Max windows per house | 100,000 |


### LSTM

| Parameter | Value |
|---|---|
| Conv1D filters / kernel size | 16 / 4 |
| BiLSTM hidden units | 128, 256 |
| Fully connected units | 128 (tanh) |
| Optimizer | Adam, no weight decay |
| Learning rate | 5e-4 |
| Batch size | 256 |
| Gradient clipping (max norm) | 1.0 |
| Loss | MSE (normalized target) |
| Max epochs / patience | 100 / 8 |
| Max windows per house | 100,000 |
| Seeds | 42, 43, 44 |
| Parameters | 1.27 M |


## Training protocol

- Up to 100 epochs, early stopping on validation loss (patience 8), Adam, MSE loss.
- Each epoch samples up to `max_windows_per_file` windows per house and shuffles houses together, so large houses don't dominate.
- Hyperparameters are chosen by grid search on a reduced budget (50 epochs, 30,000 windows per house), then the best setting is retrained with the full budget.
- Metrics: MAE and SAE.
- Experiments are tracked in Weights & Biases.

## Results

FF = Fridge-Freezer, K = Kettle, WM = Washing Machine.

### Sanity baselines (MAE)

| Baseline | FF | K | WM |
|---|---|---|---|
| Zero prediction | 24.272 | 17.447 | 18.577 |
| Median prediction | 26.122 | 17.447 | 18.577 |

### Model comparison (mean ± std)

| Model | FF MAE | FF SAE | K MAE | K SAE | WM MAE | WM SAE |
|---|---|---|---|---|---|---|
| CNN | 21.05 ± 0.34 | 0.516 ± 0.040 | 7.16 ± 0.82 | 0.060 ± 0.035 | 14.11 ± 0.67 | 0.050 ± 0.052 |
| LSTM | 16.15 ± 1.19 | 0.392 ± 0.053 | 5.85 ± 0.59 | 0.017 ± 0.006 | 13.01 ± 2.20 | 0.154 ± 0.115 |
| BERT4NILM | – | – | – | – | – | – |
| MOMENT-small (random init) | – | – | – | – | – | – |
| MOMENT-small (frozen) | – | – | – | – | – | – |
| MOMENT-small (full fine-tuning) | – | – | – | – | – | – |
