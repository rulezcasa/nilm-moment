# TSFM4NILM: Sequence-to-Point Energy Disaggregation Using MOMENT


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


## Results

Test-set MAE (W) and SAE, mean ± std over seeds 42–46. FF = Fridge-Freezer, K = Kettle, WM = Washing Machine.

| Model | FF MAE | FF SAE | K MAE | K SAE | WM MAE | WM SAE |
|---|---|---|---|---|---|---|
| CNN | 20.68 ± 0.56 | 0.468 ± 0.072 | 6.88 ± 0.72 | 0.050 ± 0.030 | 14.01 ± 0.59 | 0.069 ± 0.047 |
| LSTM | 19.53 ± 7.75 | 0.460 ± 0.158 | 6.20 ± 0.71 | 0.018 ± 0.005 | 12.70 ± 1.85 | 0.125 ± 0.093 |
| BERT4NILM | 16.16 ± 1.21 | 0.354 ± 0.065 | 5.46 ± 0.61 | 0.086 ± 0.035 | 11.65 ± 0.84 | 0.081 ± 0.070 |
| MOMENT-small (random init) | 17.02 ± 1.72 | 0.312 ± 0.114 | 8.38 ± 3.62 | 0.146 ± 0.138 | 19.58 ± 2.21 | 0.242 ± 0.105 |
| MOMENT-small (frozen) | 36.78 ± 1.90 | 0.868 ± 0.133 | 18.17 ± 1.11 | 0.286 ± 0.080 | 28.54 ± 1.84 | 0.535 ± 0.103 |
| MOMENT-small (full fine-tuning) | 14.45 ± 0.66 | 0.301 ± 0.058 | 5.80 ± 0.17 | 0.033 ± 0.026 | 11.96 ± 0.56 | 0.120 ± 0.029 |

## Directory structure

```
nilm-moment/
├── configs/
│   ├── base.yaml           # seed, device, target appliance, model, data, W&B
│   └── models/             # per-model architecture, training and search settings
│       ├── cnn.yaml
│       ├── lstm.yaml
│       ├── bert4nilm.yaml
│       ├── moment_random.yaml
│       ├── moment_frozen.yaml
│       └── moment_finetune.yaml
├── data/                  
│   ├── cleaned_source/     # REFIT cleaned CSVs
│   ├── segmented/          # gap-segmented houses
│   └── splits/<appliance>/<train|val|test>/
├── figures/                # EDA and prediction-vs-actual figures
├── notebooks/
│   ├── EDA.ipynb           # exploratory analysis
│   ├── splits.ipynb        # segmentation and house splits
│   └── span_plots.ipynb    # prediction-vs-actual figures from spans/
├── runs/<model>/<appliance>/   # checkpoints, search results, evaluations
├── scripts/
│   ├── search.py           # grid search
│   ├── train.py            # train with selected config
│   ├── evaluate.py         # evaluate best checkpoint
│   └── predict_span.py     # best-seed predictions on one test span per appliance
├── slurm/                  # HPC job scripts: search, train, evaluate, predict_span
├── spans/<appliance>.{npz,json}  # prediction-vs-actual spans from predict_span
├── src/
│   ├── data/               # IO, windowing, sampling, normalization, datasets
│   ├── evaluation/         # metrics, evaluation runner, sanity baselines
│   ├── models/             # CNN, LSTM, BERT4NILM, MOMENT
│   ├── training/           # trainer, search, checkpointing, W&B tracking
│   └── utils/              # config loading, seeding
├── EVIDENCE_RECORD.md      # architectures, tuned configs, protocol, results
└── requirements.txt
```
The processed and segmented data (`data/`) is released at [doi:10.57967/hf/11717](https://doi.org/10.57967/hf/11717).

## Reproducibility

```bash
pip install -r requirements.txt
```

Set `target` and `model` in [configs/base.yaml](configs/base.yaml), then:

```bash
python -m scripts.search    # grid search -> runs/<model>/<appliance>/search-<name>
python -m scripts.train     # train -> runs/<model>/<appliance>/<seed>
python -m scripts.evaluate  # evaluate best.pt on the test (or val) split
python -m scripts.predict_span --target Kettle  # best-seed predictions on a test span -> spans/
```

### Training

1. **Search**: a grid search for each model and appliance on a reduced budget (50 epochs, 30,000 windows per house).
2. **Train**: the best configuration is retrained with up to 100 epochs, early stopping patience of 8 on validation loss, and seeds 42–46. For one appliance and seed.
3. **Evaluate**: the `best.pt` checkpoint of each seed is scored on the test split. Results are reported as mean ± std over seeds.

The full protocol and the tuned configurations are in [EVIDENCE_RECORD.md](EVIDENCE_RECORD.md).

#### Additional information for CS-271 submission :
- Teammate 2 evaluated the saved checkpoints over re-training the model due to computational overhead.
- Every experiment and protocol planned was verified unanimously before execution.


## Acknowledgments

H. Thangaraj and S. Janavi contributed to this work while undertaking a project for CS-271 under the guidance of Prof. Jelena Gligorijevic at San Jose State University. The authors acknowledge the Computer Science Department at San Jose State University for providing computational resources and support for this work.

## Data Access Statement

This study uses the REFIT Electrical Load Measurements dataset available at [doi:10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122) and [doi:10.15129/9ab14b0e-19ac-4279-938f-27f643078cec](https://doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec) under the CC BY 4.0 license. The processed and segmented dataset created in this study is available at [doi:10.57967/hf/11717](https://doi.org/10.57967/hf/11717).

## AI Usage Statement

AI coding tools were used to assist with writing, refactoring and debugging code. The experimental planning, choice of ablations, interpretation of results were carried out by the authors. All AI-assisted output was reviewed and verified by the authors, who take full responsibility for the content of this repository.

## Author contribution

All authors contributed equally to this project. H. Thangaraj pre-processed the dataset, implemented the training and evaluation pipeline. S. Janavi planned the experiments, implemented the models and documented findings. The authors declare no competing interest in this work.

## License

The code in this repository is released under the [MIT License](LICENSE). The REFIT data and the processed dataset remain according to the Data Access Statement.
