# Reproduction notes

## Verified paper facts

- Ten datasets: Celeba (202,599 x 39, 2.24% anomalies), ALOI (49,534 x 27, 3.04%), Skin (9,521 x 3, 3.67%), Mnist (7,603 x 100, 9.21%), FashionMNIST (6,315 x 512, 5.00%), CIFAR10 (5,263 x 512, 5.00%), Waveform (3,505 x 21, 4.62%), InternetAds (1,966 x 1,555, 18.72%), Letter (1,600 x 32, 6.25%), and Vowel (1,456 x 12, 3.43%).
- Five clients and one edge server.
- Each dataset is randomly divided into five client datasets.
- Subsample size 512; OptIForest cut threshold 403.
- All experiments are repeated 20 times and report mean +/- standard deviation.
- Metrics: AUC-ROC and AUC-PR.
- Algorithm 1 trains trees layer by layer: clients choose a local feature and cut, send parameters and local sizes to the edge server, receive a weighted global feature/cut, then partition locally.

## What is not fully specified or not available

1. The paper gives no public FLiForest source URL, random seeds, number of trees, exact preprocessed matrix files, class mappings, splitting protocol, or the full OptIForest sampling/selection integration.
2. The cited UCI and ADRepository sources do not by themselves identify immutable versions for every listed matrix. The cloned ADRepository includes Celeba and InternetAds/AD-related source data, but not the paper's final prepared bundle.
3. Accordingly, `results/paper_reported_auc_roc.csv` is a transcription of Table 3, not a generated result. Values are percentages. A run through `scripts/run_experiment.py` writes `source=computed` and AUC in [0, 1].
4. The supplied model does not transmit raw data. It does transmit randomized local candidate cuts and local node counts, exactly the type of parameter exchange assumed by the paper. This is not a formal privacy guarantee; secure aggregation, differential privacy, and adversarial protections are outside the paper algorithm and this reproduction.

## Referenced repositories cloned in this workspace

| Repository | Pinned checked-out commit | Paper purpose |
| --- | --- | --- |
| `third_party/LSHiForest` | `8644f68` | L2SH baseline |
| `third_party/OptIForest` | `a8be90d` | FLiForest's iForest extension / threshold reference |
| `third_party/deep-iforest` | `e77692b` | DiForest baseline |
| `third_party/ADRepository-Anomaly-detection-datasets` | `7719051` | Dataset source |

These repositories are third-party snapshots. Do not edit them as part of this reproduction; see their individual licenses and READMEs.

## Graph provenance

`outputs/paper_reported_auc_roc.png` visualizes only the paper-reported iForest and FLiForest cells from Table 3. It must not be presented as a new experimental measurement. Use `scripts/run_all_local.py` after supplying all prepared matrices to create a computed comparison graph.
