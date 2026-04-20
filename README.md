# 🌌 Stellar Classification — SDSS DR17

Supervised machine learning project classifying astronomical objects from the Sloan Digital Sky Survey (SDSS DR17) into three classes: **STAR**, **GALAXY**, and **QSO** (quasi-stellar objects / quasars).

The central research question: *how much does the spectroscopic `redshift` parameter contribute to classification quality, and what happens when it is removed?*

---

## Project Structure

```
project/
├── 01_eda.ipynb                     # Exploratory Data Analysis
├── 02_preprocessing.ipynb           # Feature engineering & train/test split
├── 03_models.ipynb                  # Model training & evaluation
├── 04_experiment_no_redshift.ipynb  # Photometric-only experiment (formaly non-existent, inclused into 03)
├── 05_feature_importance.ipynb      # Feature importance analysis (must be 04)
└── output/
    ├── dataset_full.npz             # Full feature set (with redshift)
    ├── dataset_no_redshift.npz      # Photometric-only feature set
    ├── rf_full.pkl                  # Trained RF model (full)
    ├── rf_noz.pkl                   # Trained RF model (no redshift)
    ├── metrics.json                 # Accuracy & F1 for all models
    ├── feature_importances.json     # RF feature importances (both models)
    ├── report_knn_full.txt          # Classification report — KNN (full)
    ├── report_rf_full.txt           # Classification report — RF (full)
    ├── report_rf_no_redshift.txt    # Classification report — RF (no redshift)
    ├── fi_summary.txt               # Feature importances summary
    ├── eda_01_class_distribution.png
    ├── eda_02_violin_filters.png
    ├── eda_03_boxplots_filters.png
    ├── eda_04_redshift.png
    ├── eda_05_correlation_matrix.png
    ├── eda_06_colour_diagrams.png
    ├── eda_07_sky_map.png
    ├── eda_08_tsne.png
    ├── eda_09_umap.png
    ├── models_01_confusion_matrices.png
    ├── models_02_comparison.png
    ├── fi_01_full.png
    ├── fi_02_noz.png
    └── fi_03_comparison.png
```

---

## Dataset

- **Source:** [SDSS DR17](https://www.sdss.org/dr17/) — Sloan Digital Sky Survey, Data Release 17
- **Size:** 100 000 objects (balanced sampling across classes)
- **Target classes:** STAR (0), GALAXY (1), QSO (2)
- **Class distribution:** ~21.6% STAR · ~59.4% GALAXY · ~19.0% QSO
- **Train/test split:** 80% / 20%, stratified by class, `random_state=42`

### Feature Sets

| Feature | Type | Description |
|---|---|---|
| `u` | Photometric | Ultraviolet filter magnitude |
| `g` | Photometric | Green filter magnitude |
| `r` | Photometric | Red filter magnitude |
| `i` | Photometric | Near-infrared filter magnitude |
| `z` | Photometric | Infrared filter magnitude |
| `alpha` | Coordinate | Right ascension (sky coordinate) |
| `delta` | Coordinate | Declination (sky coordinate) |
| `redshift` | Spectroscopic | Cosmological redshift *(excluded in no-redshift experiment)* |

Two feature sets were used:
- **Full** (`FEATURES_FULL`): all 8 features including `redshift`
- **Photometric-only** (`FEATURES_NOZ`): 7 features, `redshift` excluded

---

## Pipeline Architecture

### Key Functions (`02_preprocessing.ipynb`)

```python
PHOTOMETRIC_FEATURES = ["u", "g", "r", "i", "z"]
COORD_FEATURES       = ["alpha", "delta"]
SPECTRAL_FEATURES    = ["redshift"]
TARGET_COLUMN        = "class"
TARGET_MAPPING       = {"STAR": 0, "GALAXY": 1, "QSO": 2}
```

**`get_feature_list(include_redshift: bool)`**
Returns the ordered list of input features. When `include_redshift=False`, returns only photometric + coordinate features.

**`prepare_train_test(df, include_redshift, test_size, random_state)`**
Full end-to-end preparation:
1. Calls `get_feature_list()` to select features
2. Calls `encode_target()` to map class labels to integers
3. Performs stratified `train_test_split` (preserves class balance)
4. Calls `build_pipeline()` to create a sklearn Pipeline with StandardScaler
5. Fits the pipeline on train data, transforms both train and test
6. Returns `X_train`, `X_test`, `y_train`, `y_test` (all as NumPy arrays), and the fitted `pipeline`

**`build_pipeline(include_redshift: bool)`**
Constructs a `sklearn.pipeline.Pipeline` containing a `StandardScaler`. Ensures all features are on the same scale before model training.

**`encode_target(df)`**
Maps the string `class` column to integers using `TARGET_MAPPING`. Returns a pandas Series.

**`load_data(filepath)`**
Loads the SDSS CSV file into a clean DataFrame. Compatible with both DR17 and DR18 column naming conventions.

### Datasets saved (`output/`)

| File | Contents |
|---|---|
| `dataset_full.npz` | `X_train`, `X_test`, `y_train`, `y_test` — scaled, full feature set |
| `dataset_no_redshift.npz` | Same splits, 7 photometric+coordinate features only |

Both saved with `np.savez_compressed` for efficient storage and fast loading across notebooks.

### Models saved

Models are serialized with `joblib` after training in `03_models.ipynb`:

```python
joblib.dump(rf_full, OUTPUT_DIR / 'rf_full.pkl')
joblib.dump(rf_noz,  OUTPUT_DIR / 'rf_noz.pkl')
```

This allows `05_feature_importance.ipynb` (and any future notebooks) to load pre-trained models without re-running `03_models.ipynb`.

---

## Models

Three models were trained and evaluated:

| Model | Feature set | Notes |
|---|---|---|
| KNN (full) | Full (8 features) | K-Nearest Neighbours baseline |
| RF (full) | Full (8 features) | Random Forest, primary model |
| RF (no redshift) | Photometric-only (7 features) | Ablation study — no spectroscopy |

**Shared RF hyperparameters (`RF_PARAMS`)** were used for both RF models to ensure fair comparison.

### `evaluate_model(model, X_test, y_test, label)`

Utility function used in `03_models.ipynb` for consistent evaluation. Returns a dict containing:
- `accuracy` — overall accuracy on the test set
- `f1_weighted` — weighted F1 score
- `report` — full `sklearn.metrics.classification_report` string (per-class precision, recall, F1)
- `y_pred` — model predictions on the test set
- `label` — model name string

---

## Results

### Overall Metrics

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| KNN (full) | 0.9382 | 0.9383 |
| RF (full) | **0.9778** | **0.9777** |
| RF (no redshift) | 0.8787 | 0.8779 |

### Per-Class Results — RF (full)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| STAR | 0.99 | 1.00 | 0.99 | 4 319 |
| GALAXY | 0.98 | 0.98 | 0.98 | 11 889 |
| QSO | 0.96 | 0.94 | 0.95 | 3 792 |
| **macro avg** | **0.98** | **0.97** | **0.97** | 20 000 |

### Per-Class Results — RF (no redshift)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| STAR | 0.83 | 0.76 | 0.80 | 4 319 |
| GALAXY | 0.92 | 0.94 | 0.93 | 11 889 |
| QSO | 0.79 | 0.82 | 0.81 | 3 792 |
| **macro avg** | **0.85** | **0.84** | **0.84** | 20 000 |

### Per-Class Results — KNN (full)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| STAR | 0.89 | 0.92 | 0.91 | 4 319 |
| GALAXY | 0.95 | 0.95 | 0.95 | 11 889 |
| QSO | 0.97 | 0.91 | 0.94 | 3 792 |
| **macro avg** | **0.93** | **0.93** | **0.93** | 20 000 |

---

## Feature Importance Analysis

### RF (full) — with redshift

| Feature | Importance |
|---|---|
| `redshift` | **62.0%** |
| `z` | 9.5% |
| `g` | 8.1% |
| `u` | 7.2% |
| `i` | 5.9% |
| `r` | 5.6% |
| `alpha` | 0.9% |
| `delta` | 0.8% |

### RF (no redshift) — photometric only

| Feature | Importance |
|---|---|
| `z` | 20.6% |
| `u` | 19.5% |
| `g` | 18.5% |
| `r` | 14.7% |
| `i` | 14.2% |
| `alpha` | 6.6% |
| `delta` | 5.8% |

---

## Analysis & Key Findings

### 1. Redshift dominates classification

`redshift` accounts for **62% of feature importance** in the full model — more than all other 7 features combined. This is consistent with the physical interpretation: redshift directly encodes the recession velocity of an object, and QSOs have systematically higher redshifts than stars or galaxies at similar apparent magnitudes. The single spectroscopic measurement does the heavy lifting that photometry can only partially replicate.

This finding was anticipated as early as `01_eda.ipynb`, where the redshift distribution plots showed near-complete class separation for QSOs, and was visually confirmed in t-SNE and UMAP projections: with redshift, the three classes form clean, well-separated clusters; without it, STAR and QSO clusters overlap significantly.

### 2. Without redshift, the model redistributes importance equally across photometric bands

In the no-redshift model, no single feature dominates. Importance is spread roughly evenly: `z` (20.6%), `u` (19.5%), `g` (18.5%), `r` (14.7%), `i` (14.2%). The infrared (`z`) and ultraviolet (`u`) bands emerge as slightly more informative — physically, these capture the ends of the visible spectrum where QSOs and stars diverge most due to their different spectral energy distributions.

### 3. Removing redshift degrades STAR and QSO performance most severely

GALAXY classification is largely unaffected by removing redshift (F1: 0.98 → 0.93), because galaxies have distinctive morphological signatures that translate into characteristic photometric colours. But STAR recall drops from 1.00 to 0.76, and QSO precision drops from 0.96 to 0.79. These are the two classes most easily confused without spectroscopy — a blue quasar at low redshift is photometrically almost indistinguishable from a hot blue star.

### 4. Sky coordinates (alpha, delta) contribute minimally

`alpha` and `delta` carry only 0.9% / 0.8% importance in the full model. In the no-redshift model they rise to 6.6% / 5.8% — but only because the model compensates for the absence of redshift by latching onto any available signal. The sky map in `01_eda.ipynb` confirms that all three classes are distributed uniformly across the observed sky with no meaningful spatial separation. Including coordinates does not harm the model, but they are not genuinely informative features.

### 5. RF significantly outperforms KNN

RF (full) achieves 0.978 accuracy vs KNN's 0.938 — a meaningful gap. Random Forests handle non-linear boundaries and feature interactions more effectively than the distance-based KNN approach. For this dataset, the strong dominance of `redshift` also benefits RF, which can weight features via impurity reduction; KNN treats all features equally unless explicitly scaled differently.

---

## Visualisations

All charts are saved to `output/` and generated by their respective notebooks.

| File | Description | Notebook |
|---|---|---|
| `eda_01_class_distribution.png` | Bar chart of class balance (STAR / GALAXY / QSO) | `01_eda` |
| `eda_02_violin_filters.png` | Violin plots of u/g/r/i/z magnitudes per class | `01_eda` |
| `eda_03_boxplots_filters.png` | Boxplots of filter magnitudes per class | `01_eda` |
| `eda_04_redshift.png` | Redshift distribution per class — key discriminator visible | `01_eda` |
| `eda_05_correlation_matrix.png` | Pearson correlation heatmap of all features | `01_eda` |
| `eda_06_colour_diagrams.png` | Colour-colour diagrams (g-r vs r-i, u-g vs g-r) | `01_eda` |
| `eda_07_sky_map.png` | RA/Dec scatter plot of all objects by class | `01_eda` |
| `eda_08_tsne.png` | t-SNE 2D projection (with and without redshift) | `01_eda` |
| `eda_09_umap.png` | UMAP 2D projection (with and without redshift) | `01_eda` |
| `models_01_confusion_matrices.png` | Confusion matrices for all 3 models | `03_models` |
| `models_02_comparison.png` | Accuracy & F1 bar comparison across models | `03_models` |
| `fi_01_full.png` | Horizontal bar chart — RF (full) feature importances | `05_feature_importance` |
| `fi_02_noz.png` | Horizontal bar chart — RF (no redshift) importances | `05_feature_importance` |
| `fi_03_comparison.png` | Side-by-side photometric feature comparison | `05_feature_importance` |

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn umap-learn joblib

# 2. Run notebooks in order
jupyter notebook 01_eda.ipynb
jupyter notebook 02_preprocessing.ipynb
jupyter notebook 03_models.ipynb
jupyter notebook 04_experiment_no_redshift.ipynb
jupyter notebook 05_feature_importance.ipynb
```

All outputs are saved to `output/`. Each notebook is self-contained and loads data from the files generated by the previous step.

---

## Dataset Citation

Abdurrouf et al. (2022), *The Seventeenth Data Release of the Sloan Digital Sky Surveys: Complete Release of MaNGA, MaStar, and APOGEE-2 Data*, ApJS, 259, 35. [doi:10.3847/1538-4365/ac4414](https://doi.org/10.3847/1538-4365/ac4414)
