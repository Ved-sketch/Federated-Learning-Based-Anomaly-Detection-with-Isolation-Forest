# FLiForest reproduction workspace

This project reproduces the *protocol* in Xiang et al., **Federated Learning-Based Anomaly Detection with Isolation Forest in the IoT-Edge Continuum** (ACM TOMM, 2026), and provides a runnable comparison between a centralized Isolation Forest and an FLiForest-style federated Isolation Forest.

## What is reproduced

| Paper setting | This project |
| --- | --- |
| Five clients and one edge server | Five local partitions coordinated by an in-process edge server |
| Sampling size 512 | `--subsample-size 512` |
| Twenty experiments | `--repetitions 20` |
| Layer-by-layer federated tree construction | Each active client proposes a feature and split using only its local node data; the edge server chooses a weighted global feature and a weighted global split threshold |
| iForest score | `2 ** (-mean_path_length / c(psi))` |
| AUC-ROC | mean and sample standard deviation across repetitions |

The paper calls its instantiation **FLiForest** and uses OptIForest's sampling procedure (plus a cut threshold of 403). Its full released FLiForest implementation and the exact preprocessed ten-dataset bundle were not cited or located in the repositories named by the paper. Consequently, this project is a **faithful protocol implementation, not a claim of bit-for-bit reproduction**. The provided experiment comparison is deliberately standard iForest versus the generic binary FLiForest tree specified by Algorithm 1. It does not reimplement OptIForest's proprietary/experimental tree selection procedure.

## Paper datasets and availability

The paper's Table 2 names: Celeba, ALOI, Skin, Mnist, FashionMNIST, CIFAR10, Waveform, InternetAds, Letter, and Vowel. It says they come from UCI and ADRepository, after missing-value removal and correction of wrong numbers. Only the source repositories are cited; dataset version files, label choices, and preprocessing scripts are not.

`third_party/ADRepository-Anomaly-detection-datasets` is cloned locally and contains the cited Celeba source and an InternetAds/AD source, but it does **not** contain all of the paper's final 10 prepared matrices. Place prepared matrices under `data/local/` and pass each to the runner. See `docs/REPRODUCTION_NOTES.md` for the exact table, repo revisions, and limitations.

## Quick start

Create a virtual environment and install the small runtime dependency set:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Smoke-test the complete training/evaluation path on a deterministic synthetic dataset:

```powershell
python scripts/run_experiment.py --synthetic --repetitions 2 --n-trees 50 --output results/smoke.csv
```

Run one of your prepared matrices. The loader accepts `.npz` files with `X` and `y` arrays, or CSV files with a `label` column (or one selected by `--label-column`). Labels must use `1` for anomaly and `0` for normal.

```powershell
python scripts/run_experiment.py --dataset-file data/local/waveform.npz --dataset-name Waveform --repetitions 20 --subsample-size 512 --clients 5 --n-trees 100 --output results/waveform_run.csv
python scripts/plot_results.py --input results/waveform_run.csv --output outputs/waveform_auc_roc.png
```

To reproduce the chart of the values reported in the paper (not newly computed values), use:

```powershell
python scripts/plot_results.py --input results/paper_reported_auc_roc.csv --output outputs/paper_reported_auc_roc.png --title "Paper-reported AUC-ROC: iForest vs FLiForest"
```

## Running all local datasets

Put one `.npz` file per dataset in `data/local/`, using exactly these stems: `celeba`, `aloi`, `skin`, `mnist`, `fashionmnist`, `cifar10`, `waveform`, `internetads`, `letter`, and `vowel`. Then run:

```powershell
python scripts/run_all_local.py --data-dir data/local --repetitions 20 --subsample-size 512 --clients 5 --n-trees 100
```

This writes a computed `results/computed_auc_roc.csv` and `outputs/computed_auc_roc.png`. The runner uses a single 80/20 stratified split for each repetition and only trains on the training split; this split choice is an explicit approximation because the paper does not state a train/test partition protocol.

## Referenced repositories

The following repositories have been cloned under `third_party/` at the commits recorded in `docs/REPRODUCTION_NOTES.md`:

- [LSHiForest](https://github.com/xuyun-zhang/LSHiForest)
- [OptIForest](https://github.com/xiagll/OptIForest)
- [Deep Isolation Forest](https://github.com/xuhongzuo/deep-iforest)
- [ADRepository anomaly-detection datasets](https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets)

They are kept separate from this implementation and are not imported at runtime. Their own licenses and requirements apply.

## Outputs already included

- `results/paper_reported_auc_roc.csv`: Table 3 iForest and FLiForest mean/std values transcribed from the paper, in percent.
- `outputs/paper_reported_auc_roc.png`: a graph generated from that CSV. It is clearly labelled as paper-reported.
- `results/smoke.csv`: generated only when you run the smoke command above; computed files are always marked `source=computed`.

