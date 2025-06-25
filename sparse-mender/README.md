# Sparse-MENDER (SMENDER)

**Sparse-MENDER** explores scalable variants of the original [**MENDER**](https://doi.org/10.1038/s41467-023-44367-9) algorithm by integrating **approximate nearest neighbors (ANN)** and **alternative linear dimensionality reduction techniques** to improve performance and efficiency on large spatial omics datasets.

> **MENDER: Fast and scalable tissue structure identification in spatial omics data**  
> Zhiyuan Yuan et al., Nature Communications, 2024  
> [DOI: 10.1038/s41467-023-44367-9](https://doi.org/10.1038/s41467-023-44367-9)

The goal is to explore the possibility of replacing **exact k-NN** with **Approximate Nearest Neighbors (ANN)** techniques:
  * [Annoy](https://github.com/spotify/annoy)
  * [Hierarchical Navigable Small Worlds (HNSW)](https://github.com/nmslib/hnswlib)

Additionally, we also explore replacing the default [Principal Components Analysis (PCA)](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html) with alternative linear dimensionality reduction techniques:
  * [Non-negative Matrix Factorization (NMF)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
  * [Independent Component Analysis (ICA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
  * [Factor Analysis (FA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html)

We conduct the experiment by tracking and verifing clustering quality and efficiency metrics, with statistical significance testing.

---

## Notebooks

This project includes multiple Jupyter notebooks under `experiments/`:

| Notebook                | Dataset            | Description                                                                                                                                                   |
| ----------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `smender-merfish.ipynb` | MERFISH            | Runs Sparse-MENDER with configurable ANN + dimensionality reduction                                                                                           |
| `smender-visium.ipynb`  | Visium MOB         | Same as above, adapted for Visium  MOB dataset                                                                                             |
| `analysis.ipynb`        | Aggregated results | Performs statistical testing (e.g., t-tests) and generates plots (e.g., box plots, bar plots) to validate significance of results |

> 💡 All experimental notebooks (`smender-merfish.ipynb` and `smender-visium.ipynb`) allow configuration of:
>
> * Neighbor search: `None` (exact), `AnnoyANN`, `HNSWANN`
> * Linear dimensionality reduction: `'PCA'`, `'NMF'`, `'ICA'`, `'FA'`

---

## Metrics Tracked

### Clustering Quality

| Metric    | Description                                                    |
| --------- | -------------------------------------------------------------- |
| **ARI**   | Adjusted Rand Index — agreement with ground truth              |
| **NMI**   | Normalized Mutual Information — cluster overlap                |
| **PAS**   | Percentage of Agreement Score — neighborhood label consistency |
| **CHAOS** | Measures fragmentation of clusters — lower is better           |

Metrics are implemented in [`utils.py`](./smender/utils.py).

### Efficiency Profiling

| Component                | Metrics                                                          | Description                                  |
| ------------------------ | ---------------------------------------------------------------- | -------------------------------------------- |
| Full SMENDER run         | Total runtime & memory                                           | `smender_time`, `smender_memory`             |
| Dimensionality reduction | Time & memory usage for only linear dimensionality reduction     | `dim_reduction_time`, `dim_reduction_memory` |
| Neighbor search          | ime & memory for only graph construction (exact or approximate). | `nn_time`, `nn_memory`                       |

---

## Key Classes

| Class            | Location             | Purpose                                                                                                              |
| ---------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `SMENDER`        | `smender/SMENDER.py` | Controls multi-batch clustering pipeline, metric tracking, and output management                                     |
| `SMENDER_single` | `smender/SMENDER.py` | Executes clustering for one batch of spatial omics data                                                              |
| `ANN`            | `ann/ANN.py`         | Base interface for approximate neighbor implementations                                                              |
| `AnnoyANN`       | `ann/AnnoyANN.py`    | ANN graph construction using Annoy                                                                                   |
| `HNSWANN`        | `ann/HNSWANN.py`     | ANN graph construction using HNSWLib                                                                                 |
| `DataLoader`     | `data_loader.py`     | Loads datasets from SODB                                                                                             |
| Metrics          | `utils.py`           | Functions like `compute_ARI`, `compute_NMI`, `compute_PAS`, `compute_CHAOS`, and `res_search` for cluster validation |

---

## Setup Instructions

```bash
# Navigate to sparse-mender directory if at root of project
cd sparse-mender

# Setup virtual environment
python -m venv smender-venv

# Activate virtual environment
source smender-venv/Scripts/activate  # On Windows

# Install dependencies
pip install -r smender-requirements.txt
```

---

## Running Experiments

Configure:

* `ann` as `None`, `AnnoyANN`, or `HNSWANN`
* `dim_reduction` as `'pca'`, `'nmf'`, `'ica'`, or `'fa'`
* Number of clusters, number of scales, etc.

Use the following for experiments on the MERFISH dataset:

```bash
jupyter notebook experiments/smender-merfish.ipynb
```

Use the following for experiments on the Visium MOB dataset:

```bash
jupyter notebook experiments/smender-visium.ipynb
```

To analyze results:

```bash
jupyter notebook experiments/analysis.ipynb
```

---

## Author

**Inaesh Joshi**
> [i.joshi-2@student.tudelft.nl](mailto:i.joshi-2@student.tudelft.nl)<br>
> MSc Data Science and Artificial Intelligence Technologies<br>
> Delft University of Technology<br>
> Delft, Netherlands

## AI Disclaimer
README.md was refined using ChatGPT-4o, but all ideas and implementation belong to the author.
