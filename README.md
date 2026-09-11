# SilentWatch: A Novel Frame-Level Time-of-Action Prediction Method for Audio-Agnostic Surveillance

SilentWatch is a dual-head deep-learning architecture for predicting the
frame-level **Time of Action (ToA)** in silent surveillance video. The model
uses a lightweight MobileNetV2 spatial encoder to extract frame-level visual
features and an LSTM temporal encoder to model action progression across a
fixed-length frame sequence. Shared temporal features feed separate
classification and ToA regression heads.

The approach is designed for audio-agnostic surveillance analysis where audio
streams, scene-level labels, or continuous object tracking are unavailable or
undesirable.

## Architecture

The end-to-end pipeline is:

1. Videos are sampled at the configured rate and resized to `224 x 224`.
2. MobileNetV2 extracts a `1280`-dimensional feature vector from each frame.
3. An LSTM processes the sequence of frame embeddings.
4. Two prediction heads use the final temporal context vector:
   - **Classification head:** predicts the action class.
   - **Regression head:** predicts the continuous ToA frame index.

Training uses the weighted multi-task objective:

```text
Total Loss = 1.0 * CrossEntropy(classification) + 2.0 * MSE(ToA)
```

The default configuration uses a 16-frame sequence, Adam optimization with a
learning rate of `1e-4`, reproducible dataset splits of `70% / 15% / 15%`,
validation-ToA-MAE checkpoint selection, learning-rate reduction on plateau,
and early stopping.

## Final Paper Results

The following values are the final results reported in the manuscript and are
stored in [`metrics.json`](./metrics.json):

| Metric | Result |
|---|---:|
| Classification accuracy | 78.54% |
| Classification F1-score | 0.78 |
| ToA mean absolute error | 2.14 frames |
| ToA localization accuracy | 90.91% |
| Localization tolerance | +/- 5 frames |

These metrics are manuscript-reported reference results. Reproducing them
requires the evaluation data, frame-level ToA annotations, trained checkpoint,
and experiment settings used for the published run. The official SPHAR
classification dataset does not itself provide verified frame-level ToA
annotations.

## Repository Structure

```text
.
|-- metrics.json                 # Manuscript-reported final metrics
|-- auto_annotate.py             # Synthetic motion-delta ToA annotation tool
|-- Silent_watch-main/
    |-- config.yaml              # Dataset, model, and training configuration
    |-- smoke_test.py             # CPU-only forward/backward validation
    |-- predict_toa.py            # ToA and class inference utility
    |-- src/
        |-- data_loader.py        # Frame-sequence dataset and DataLoader
        |-- preprocess.py         # Video sampling, resizing, and metadata creation
        |-- train.py              # Dual-head training loop
        |-- evaluate.py           # Accuracy, F1, MAE, and localization metrics
        |-- models/
            |-- model.py          # MobileNetV2 + LSTM/GRU dual-head model
```

## Setup

Create and activate a Python virtual environment, then install the runtime
dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas opencv-python PyYAML tqdm
```

The repository was developed and smoke-tested with Python 3.11. CPU execution
is sufficient for the smoke test. A CUDA-capable PyTorch installation is
recommended for full video training.

## CPU Smoke Test

Run the architecture-only test from the project directory:

```powershell
Set-Location Silent_watch-main
python smoke_test.py
```

The smoke test creates random clips shaped `(4, 16, 3, 224, 224)`, runs them
through MobileNetV2 and the LSTM, computes the joint loss, performs
`loss.backward()`, and executes an optimizer step. It does not download or
require a dataset.

## Data Preparation

Set the dataset paths in
[`Silent_watch-main/config.yaml`](./Silent_watch-main/config.yaml), then run:

```powershell
Set-Location Silent_watch-main
python src/preprocess.py
```

For a CSV with verified annotations, provide either `toa_frame` in source-frame
coordinates or `toa_seconds` in seconds:

```yaml
dataset:
  annotation_csv: "D:/annotations/toa.csv"
```

The preprocessing step converts source-frame labels to sampled-frame
coordinates using the configured sampling rate.

For pipeline validation when verified annotations are unavailable, the
repository also includes a synthetic motion-delta annotator:

```powershell
python auto_annotate.py `
  --input-dir D:\SPHAR-subset `
  --output-csv D:\toa_auto.csv `
  --threshold 15
```

Synthetic motion-delta labels identify the first large visual change. They are
useful for exercising the regression pipeline but should not be treated as
human-verified action-onset ground truth.

## Training and Evaluation

After preprocessing creates `metadata.csv` with valid ToA labels:

```powershell
Set-Location Silent_watch-main
python src/train.py
python src/evaluate.py `
  --model_path silentwatch_best_model.pth `
  --metadata D:\processed\metadata.csv
```

To inspect predictions for a small number of clips:

```powershell
python predict_toa.py `
  --model-path silentwatch_best_model.pth `
  --metadata D:\processed\metadata.csv `
  --limit 5
```

Evaluation reports classification accuracy, macro F1, ToA MAE, and localization
accuracy within the configured tolerance window.
