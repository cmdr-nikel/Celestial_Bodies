# STELLaRUM — Observation Terminal
### Stellar Classification · SDSS DR17 · Interactive Frontend

> *A university ML project that classifies stars, galaxies, and quasars from the Sloan Digital Sky Survey — wrapped in a browser-based radio telescope game inspired by [Voices of the Void](https://mrdrnose.itch.io/votv).*

---

## Overview

STELLaRUM is a two-part project:

- **Backend** — supervised ML pipeline classifying 100,000 celestial objects (STAR / GALAXY / QSO) using KNN and Random Forest. The central experiment tests how much the spectroscopic `redshift` feature contributes to accuracy, and what happens when it is removed.
- **Frontend** — a self-contained browser game (`stella_rum.html`) where the player operates a three-dish antenna array, triangulates signals, and classifies objects using real SDSS DR17 photometric data. No server required.

---

## Project Structure

```
stellarum/
├── backend/
│   ├── 01eda.ipynb                  # Exploratory Data Analysis
│   ├── 02preprocessing.ipynb        # Feature engineering, train/test split
│   ├── 03models.ipynb               # KNN, Random Forest training
│   ├── 05featureimportance.ipynb    # RF feature importance analysis
│   └── output/
│       ├── dataset_full.npz         # Scaled features + labels, full set
│       ├── dataset_noredshift.npz   # Photometric-only feature set
│       ├── rf_full.pkl              # Trained RF model (with redshift)
│       ├── rf_noz.pkl               # Trained RF model (no redshift)
│       ├── metrics.json             # Accuracy & F1 for all models
│       ├── feature_importances.json # RF feature importances, both models
│       └── *.png                    # EDA, confusion matrix, FI charts
│
├── frontend/
│   ├── stella_rum.html              # Main entry point — the entire game
│   ├── objects.json                 # 1 500 SDSS DR17 objects for the game
│   └── ambient.mp3                  # Optional ambient audio track
│
└── README.md
```

---

## Dataset

- **Source:** SDSS DR17 — Sloan Digital Sky Survey, Data Release 17
- **Size:** 100,000 objects (balanced sampling across classes)
- **Classes:** STAR `0` · GALAXY `1` · QSO `2`
- **Distribution:** 21.6% STAR · 59.4% GALAXY · 19.0% QSO
- **Split:** 80/20 train/test, stratified by class, `random_state=42`

| Feature | Type | Description |
|---|---|---|
| `u` `g` `r` `i` `z` | Photometric | Filter magnitudes UV→infrared |
| `alpha` `delta` | Coordinate | Right ascension & declination |
| `redshift` | Spectroscopic | Cosmological redshift — key discriminator |
| `class` | Target | STAR / GALAXY / QSO |

Two feature sets were used throughout:
- **Full** — all 8 features including `redshift`
- **Photometric-only** — 7 features, `redshift` excluded

---

## Backend — ML Pipeline

### Models

| Model | Feature set | Accuracy | F1 (weighted) |
|---|---|---|---|
| KNN | Full (8 features) | 0.9382 | 0.9383 |
| Random Forest | Full (8 features) | **0.9778** | **0.9777** |
| Random Forest | No redshift (7 features) | 0.8787 | 0.8779 |

### RF Full — Per-class results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| STAR | 0.99 | 1.00 | 0.99 |
| GALAXY | 0.98 | 0.98 | 0.98 |
| QSO | 0.96 | 0.94 | 0.95 |

### Key experiment — with vs. without redshift

The central question of the backend: **how much does a single spectroscopic measurement carry the entire model?**

`redshift` accounts for **62%** of RF feature importance — more than all 7 other features combined. Removing it drops accuracy from 0.978 → 0.879, with STAR recall falling from 1.00 → 0.76 and QSO precision from 0.96 → 0.79. These are the two classes most easily confused without spectroscopy: a low-redshift quasar is photometrically almost indistinguishable from a hot blue star.

Without redshift, importance distributes evenly across photometric bands:

| Feature | Importance (no redshift) |
|---|---|
| `z` | 20.6% |
| `u` | 19.5% |
| `g` | 18.5% |
| `r` | 14.7% |
| `i` | 14.2% |

### How to reproduce

```bash
pip install numpy pandas scikit-learn matplotlib seaborn umap-learn joblib

jupyter notebook 01eda.ipynb
jupyter notebook 02preprocessing.ipynb
jupyter notebook 03models.ipynb
jupyter notebook 05featureimportance.ipynb
```

Each notebook is self-contained and loads data produced by the previous step. All outputs go to `output/`.

---

## Frontend — Observation Terminal

The frontend is a **single HTML file** — no framework, no build step, no server. Open `stella_rum.html` in a browser and it runs.

### How to play

1. **Aim dishes** — use `←→↑↓` arrow keys to steer the active dish across the sky map
2. **Switch dishes** — `TAB` cycles through A/B/C · `1` `2` `3` selects directly
3. **Triangulate** — point all three dishes at the same object until scan circles overlap and a green lock point appears
4. **Scan** — press `S` to lock the signal. The right panel shows photometric data and spectrum
5. **Classify** — decide if the signal is a STAR, GALAXY, or QSO and press the corresponding button
6. **Score** — correct classifications are logged; wrong ones too

### Controls reference

| Key | Action |
|---|---|
| `← → ↑ ↓` | Steer active dish (azimuth / elevation) |
| `TAB` | Next dish |
| `1` `2` `3` | Select dish A / B / C |
| `S` | Scan — attempt triangulation lock |
| `R` | Reset active dish to default position |
| `drag` | Pan the sky map |
| `scroll` | Zoom in/out |

### Object types

| Type | Key signature |
|---|---|
|  STAR | Peaks in `r` and `i` bands · redshift near `z ≈ 0` · compact point in preview |
|  GALAXY | Spread evenly across all `ugriz` bands · moderate redshift `z < 1` · diffuse elliptical shape |
|  QSO | Strong `u`-band excess · high redshift `z > 1`, sometimes `z > 3` · irregular glow in preview |

### Interface panels

- **Left panel** — Antenna Array: dish coordinates, signal strength bars
- **Centre** — Sky canvas: 2D equatorial projection of 1,500 SDSS DR17 objects rendered with ASCII glyphs; classified objects revealed with colour-coded symbols
- **Right panel** — three tabs:
  - `SIGNAL` — photometric readout, ugriz spectrum bars, pixel preview, classify buttons
  - `MAIL` — narrative messages delivered as classification milestones are reached
  - `HELP` — controls reference and object type guide

### Visual effects

- **VHS/CRT overlay** — scanlines, chromatic aberration (SVG filter), tracking glitch, film grain — all via pure CSS + JS canvas, no external libraries
- **Animated splash screen** — SVG illustration of RT-64 style radio telescope with blinking nav lights, star ring, twinkling star-field canvas

### Audio

The terminal uses the Web Audio API for procedural sound effects (scan sweep, triangulation lock, classification feedback, mail notification). An optional ambient track (`ambient.mp3`) fades in on first user interaction.

---

## Architecture notes

- **No dependencies** — vanilla HTML/CSS/JS only. Zero npm, zero build tools.
- **Data flow** — `objects.json` is loaded once at startup via `fetch()`. All rendering is done on a 2D Canvas with custom equatorial projection math.
- **Physics loop** — `requestAnimationFrame` drives dish movement with velocity, acceleration and drag for smooth inertial steering.
- **Scan animation** — sky objects are rendered in a left-to-right strip scan on idle, instantly on movement, giving a radar-sweep feel.

---

## What makes this project stand out

1. **Scientifically motivated experiment** — with vs. without redshift is a real astrophysical question, not arbitrary ablation
2. **Physical interpretation** — feature importance is explained in terms of spectral energy distributions, not just statistics
3. **Frontend is a game, not a dashboard** — genuine gameplay loop makes it memorable as a portfolio piece
4. **Single-file deployment** — the entire frontend runs by opening one HTML file; no server, no install

---

## Citation

Abdurrouf et al. (2022), *The Seventeenth Data Release of the Sloan Digital Sky Survey's Complete Release of MaNGA, MaStar, and APOGEE-2 Data*, ApJS, 259, 35.  
DOI: [10.3847/1538-4365/ac4414](https://doi.org/10.3847/1538-4365/ac4414)
Dataset: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data
Original: https://www.sdss4.org/dr17/ 
