# TSFM4NILM: Sequence-to-Point Energy Disaggregation Using MOMENT

> **Note:** This is ongoing work. The repository presents intermediate results, and models, configurations and findings may change.

Non-intrusive load monitoring (NILM), also called energy disaggregation, estimates the power drawn by individual appliances from a household's aggregate electricity signal. Predicting appliance-level consumption without a separate meter per device can support energy conservation, demand management and fault detection.

NILM has been tackled with classical machine-learning and deep-learning models that learn appliance-specific patterns from aggregate power. Pretrained time-series foundation models (TSFMs) are a possible alternative: their general temporal representations may capture appliance signatures, such as switching events and operating cycles, with less task-specific training.

This project compares the **MOMENT** TSFM with convolutional, recurrent and transformer baselines on the **REFIT** dataset.

## Contributions

1. **MOMENT vs. Seq2Point baselines**: evaluates whether the pretrained temporal representations of MOMENT ([Goswami et al., 2024](https://arxiv.org/abs/2402.03885)) improve disaggregation over established convolutional, recurrent and Transformer-based Seq2Point baselines.
2. **Reproducible pipeline**: household-level splits, preprocessing steps, hyperparameters and random seeds are explicitly documented and released.
3. **Robustness on REFIT**: evaluation on REFIT ([Murray et al., 2017](https://doi.org/10.1038/sdata.2016.122)), whose noise, missing readings and labelling inconsistencies make it a demanding test of model robustness. The noise is handled through dead-aggregate removal and segmentation (see [Dataset](#dataset)).

## Dataset

[REFIT Electrical Load Measurements (cleaned)](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/): 20 UK houses (House 1–21, no House 14), with aggregate mains and up to nine appliance channels per house.

- Houses 11 and 21 are excluded because solar panels affect their aggregates.
- Rows where the appliance sum exceeds the aggregate (mostly dead aggregate readings) are removed.
- Recordings are split into segments wherever the sampling gap exceeds 2 min, so no input window spans a long gap.

## Target appliances

Each appliance has its own house-level train/val/test split (see [notebooks/splits.ipynb](notebooks/splits.ipynb)). No house appears in more than one role for the same appliance.

| Appliance | On threshold | Train houses | Val | Test houses |
|---|---|---|---|---|
| Kettle | 2000 W | 3, 6, 7, 8, 9, 12, 13, 19, 20 | 5 | 2, 15, 17 |
| Fridge-Freezer | 50 W | 2, 5, 9 | 12 | 15 |
| Washing Machine | 20 W | 2, 5, 7, 9, 15, 16, 17 | 1 | 3, 6, 8, 10, 13, 19, 20 |

## Models

All models are formulated for sequence-to-point prediction : given a window of 512 aggregate samples, they predict the target appliance's power at the window midpoint. Inputs and targets are z-score normalized with statistics from the training split.

Baselines, architectures, tuned configurations, training protocol and results are in [EVIDENCE_RECORD.md](EVIDENCE_RECORD.md).


## Directory structure

```
nilm-moment/
├── configs/
│   ├── base.yaml           # seed, device, target appliance, model, data, W&B
│   └── models/             # per-model architecture, training and search settings
│       ├── cnn.yaml
│       └── lstm.yaml
├── data/                  
│   ├── cleaned_source/     # REFIT cleaned CSVs
│   ├── segmented/          # gap-segmented houses
│   └── splits/<appliance>/<train|val|test>/
├── notebooks/
│   ├── EDA.ipynb           # exploratory analysis
│   └── splits.ipynb        # segmentation and house splits
├── runs/<model>/<appliance>/   # checkpoints, search results, evaluations
├── scripts/
│   ├── search.py           # grid search
│   ├── train.py            # train with selected config
│   └── evaluate.py         # evaluate best checkpoint
├── src/
│   ├── data/               # IO, windowing, sampling, normalization, datasets
│   ├── evaluation/         # metrics, evaluation runner, sanity baselines
│   ├── models/             # CNN, LSTM
│   ├── training/           # trainer, search, checkpointing, W&B tracking
│   └── utils/              # config loading, seeding
└── requirements.txt
```
**(`data/`) will be released separately at : {yet to publish} .**

## Usage

```bash
pip install -r requirements.txt
```

Set `target` and `model` in [configs/base.yaml](configs/base.yaml), then:

```bash
python -m scripts.search    # grid search -> runs/<model>/<appliance>/search-<name>
python -m scripts.train     # train -> runs/<model>/<appliance>/<seed>
python -m scripts.evaluate  # evaluate best.pt on the test (or val) split
```

## Acknowledgements
H. Thangaraj and S. Janavi are contributing to this project for CS-271 (Topics in Machine Learning) under Prof.Jelena Gligorijevic at the Computer Science Department, San Jose State Univeristy.


