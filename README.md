# Palimpsest
### End-to-End Satellite Change Detection via Deep Learning on Sentinel-2

> *A palimpsest is a manuscript where old writing has been scraped away and new writing placed over it — yet traces of the original remain visible. This system does exactly that: reveals what has changed between two observations of the same ground, against the backdrop of what was there before.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Model Architecture](#3-model-architecture)
4. [Training on LEVIR-CD](#4-training-on-levir-cd)
5. [Fine-Tuning on OSCD (Sentinel-2)](#5-fine-tuning-on-oscd-sentinel-2)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Earth Engine Retrieval Service](#7-earth-engine-retrieval-service)
8. [Postprocessing and Vectorisation](#8-postprocessing-and-vectorisation)
9. [REST API](#9-rest-api)
10. [AI Analysis Agent](#10-ai-analysis-agent)
11. [GCP Infrastructure](#11-gcp-infrastructure)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [Demo Interface](#13-demo-interface)
14. [Performance Summary](#14-performance-summary)
15. [Configuration Reference](#15-configuration-reference)

---

## 1. System Overview

Palimpsest is a production-grade change detection pipeline for multispectral Sentinel-2 imagery. Given a geographic bounding box and two target dates, the system:

1. Queries Google Earth Engine for the least-cloudy available Sentinel-2 scenes around each date.
2. Exports the selected scenes (bands B2/B3/B4/B8/SCL, 10 m/px) to Cloud Storage as Cloud-Optimised GeoTIFFs.
3. Applies joint percentile normalisation and cloud masking via the Scene Classification Layer (SCL).
4. Runs patch-based inference with a Siamese EfficientNet-B2 change detection model — fine-tuned on real Sentinel-2 imagery at native resolution.
5. Cleans the binary mask morphologically, vectorises change regions to GeoJSON polygons, and reprojects them to WGS-84 for storage.
6. Persists results to BigQuery (GEOGRAPHY column) and the binary mask GeoTIFF to Cloud Storage.
7. Returns a structured JSON response including change area (m²), change percentage, polygon list, and GCS URIs.
8. **(Optional) AI interpretation**: an agentic post-processing step uses Google Gemini with function calling — reverse-geocoding the change centroid and searching external knowledge bases — to produce a natural-language explanation of *what* changed and *why* (e.g. linking a detected change footprint to the 2020 Beirut port explosion).

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Client (browser / API consumer)                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ POST /api/v1/detect
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Cloud Run — FastAPI (uvicorn)                                   │
│  palimpsest-api  rev 00009+  │  2 vCPU · 4 GiB · timeout 3600s  │
│                                                                 │
│  ┌───────────────┐   ┌──────────────────┐   ┌───────────────┐  │
│  │ EE Retrieval  │   │  Preprocessor    │   │  Detector     │  │
│  │ S2_SR_HARM.   │→  │  joint-pct norm  │→  │  ChangeFormer │  │
│  │ cloud filter  │   │  SCL masking     │   │  EfficientB2  │  │
│  │ GCS export    │   │  patch extraction│   │  patch+TTA    │  │
│  └───────────────┘   └──────────────────┘   └───────┬───────┘  │
│                                                      │          │
│  ┌──────────────────────────────────────────────────┘          │
│  │  Postprocessor                                              │
│  │  binary closing · component filtering · rasterio.shapes    │
│  │  pyproj reproject UTM→WGS-84 · shapely union              │
│  └──────────────────────────────────┬───────────────────────── │
│                                     │                          │
│           ┌─────────────────────────┼──────────────────┐      │
│           ▼                         ▼                    ▼      │
│      BigQuery                  GCS (mask.tif)      API Response │
│  change_detections             results/              JSON       │
└─────────────────────────────────────────────────────────────────┘
                           │
                    Cloud Build CI/CD
                    (Artifact Registry → Cloud Run deploy)
```

### Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Satellite imagery | Google Earth Engine Python API | `S2_SR_HARMONIZED`, least-cloudy scene selection |
| Geospatial I/O | GDAL, rasterio, shapely, pyproj | Raster read/write, vectorisation, CRS reprojection |
| Change detection | Custom ChangeFormer (EfficientNet-B2 Siamese) | 7.9 M parameters, 4-channel input (BGRNIR) |
| Deep learning | PyTorch 2.x + timm | Training, inference, TTA |
| Backend API | FastAPI + Uvicorn | Async, Pydantic validation |
| Raw imagery storage | GCP Cloud Storage | COG GeoTIFF, 90-day lifecycle on raw/ prefix |
| Results storage | GCP BigQuery | GEOGRAPHY column for spatial queries |
| Container | Docker (python:3.12-slim + GDAL) | Same image for dev and production |
| CI/CD | GCP Cloud Build | 3-step: download checkpoint → build image → deploy |
| Production serving | GCP Cloud Run | Serverless, min-instances=0, max-instances=3 |
| AI analysis agent | Google Gemini (REST, function calling) | Post-detection natural-language interpretation |
| Demo UI | Leaflet.js (vanilla JS, zero framework) | Served as static files by FastAPI |

### Walkthrough

The end-to-end flow, from drawing an area of interest to an AI-generated interpretation of the detected change:

| | |
|:---:|:---:|
| ![Initial interface — world view](imagenes/Screenshot%202026-06-02%20221047.png) | ![Area of interest over Beirut port](imagenes/Screenshot%202026-06-02%20221101.png) |
| **1. Initial interface** — pick a demo preset or draw a bounding box. | **2. Area of interest** — Beirut port, before/after dates set. |
| ![Detection running](imagenes/Screenshot%202026-06-02%20221151.png) | ![Detection results with change overlay](imagenes/Screenshot%202026-06-02%20221431.png) |
| **3. Detection in progress** — Earth Engine retrieval and model inference. | **4. Results** — change polygons overlaid; 1.75 km² changed, 33 polygons. |
| ![Analysis running](imagenes/Screenshot%202026-06-02%20221446.png) | ![AI analysis card](imagenes/Screenshot%202026-06-02%20221504.png) |
| **5. AI analysis** — the agent geocodes and researches the area. | **6. Interpretation** — change correctly attributed to the Aug 2020 Beirut port explosion, HIGH confidence. |

---

## 2. Repository Structure

```
palimpsest/
├── app/
│   ├── core/
│   │   ├── config.py           # Pydantic Settings — all config from env vars
│   │   └── logging.py
│   ├── routes/
│   │   ├── detect.py           # POST /api/v1/detect  (full pipeline orchestration)
│   │   └── analyse.py          # POST /api/v1/detect/{id}/analyse  (Gemini agent)
│   ├── schemas/
│   │   ├── request.py          # DetectRequest (bbox, dates, cloud %)
│   │   └── response.py         # DetectResponse (area, pct, polygons, GCS URIs)
│   ├── services/
│   │   ├── detector.py         # ChangeFormer model + ChangeDetector inference wrapper
│   │   ├── earth_engine.py     # Sentinel-2 retrieval and GCS export
│   │   ├── postprocessor.py    # Morphological cleanup, vectorisation, statistics
│   │   ├── preprocessor.py     # Normalisation, cloud masking, patch extraction
│   │   ├── analyst.py          # ChangeAnalyst — Gemini agent (geocode + research)
│   │   └── storage.py          # GCS download/upload, BigQuery insert/query
│   └── main.py                 # FastAPI lifespan, /demo static mount, /health
│
├── model/
│   ├── dataset.py              # LEVIRDataset — PNG pairs → 4-channel float32 tensors
│   ├── oscd_dataset.py         # OSCDDataset — Sentinel-2 GeoTIFF pairs, RAM cache
│   ├── train.py                # LEVIR-CD fine-tuning loop
│   ├── evaluate.py             # Full evaluation (predict_large) + threshold sweep
│   └── colab_oscd.ipynb        # Self-contained Colab notebook for OSCD fine-tuning
│
├── demo/
│   └── map/
│       ├── index.html          # Leaflet map + sidebar
│       ├── app.js              # Demo presets, bbox draw, API calls, results rendering
│       └── style.css
│
├── infra/
│   ├── cloudbuild.yaml         # 3-step Cloud Build: GCS pull → docker build → deploy
│   ├── setup_gcp.sh            # Idempotent GCP infra setup (buckets, BQ, IAM)
│   ├── bq_schema.json          # BigQuery change_detections table schema
│   ├── gcs_lifecycle.json      # 90-day auto-delete on raw/ prefix
│   └── smoke_test.sh           # Post-deploy health + detect sanity check
│
├── tests/
│   ├── test_phase7.py          # API integration tests (health, detect, results)
│   ├── test_preprocessor.py    # Unit tests for normalize_pair, extract_patches
│   └── test_detector.py        # Unit tests for ChangeFormer forward pass
│
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 3. Model Architecture

### 3.1 Design Philosophy

The model implements a Siamese encoder with absolute-difference feature comparison — a well-established paradigm for bitemporal change detection. The encoder is shared (identical weights) for both the before and after images, which forces the representation to be temporally invariant to seasonal and illumination differences while remaining sensitive to structural changes.

The original ChangeFormer (Bandara & Patel, IGARSS 2022) uses a MiT-B2 (Mix Transformer) backbone. This implementation substitutes **EfficientNet-B2** from timm, which provides:
- Equivalent multi-scale feature hierarchies at H/4, H/8, H/16, H/32
- Faster inference on CPU (no attention computation)
- Compatible parameter count (7.9 M vs ~25 M for MiT-B2)

### 3.2 Architecture Detail

```
Input: before (B, 4, H, W)  +  after (B, 4, H, W)
       4 channels: Blue (B2), Green (B3), Red (B4), NIR (B8)
       normalised to [0, 1] via joint percentile clipping

Shared EfficientNet-B2 encoder (pretrained=False, weights from checkpoint):
  Stage 1 → (B, 24,  H/4,  W/4 )
  Stage 2 → (B, 48,  H/8,  W/8 )
  Stage 3 → (B, 120, H/16, W/16)
  Stage 4 → (B, 352, H/32, W/32)

Per-scale absolute difference + projection:
  diff_i = |f_before_i - f_after_i|          shape: (B, C_i, H_i, W_i)
  proj_i = Conv1×1-BN-ReLU(diff_i, 256)      shape: (B, 256, H_i, W_i)
  up_i   = bilinear_upsample(proj_i, H/4)    shape: (B, 256, H/4, W/4)

Fusion:
  cat    = concat([up_1, up_2, up_3, up_4])  shape: (B, 1024, H/4, W/4)
  fuse   = Conv1×1-BN-ReLU → Conv3×3-BN-ReLU shape: (B, 128, H/4, W/4)

Output head:
  logits = Conv1×1(fuse)                     shape: (B, 1, H/4, W/4)
  logits = bilinear_upsample(logits, H)      shape: (B, 1, H, W)

Inference:
  proba  = sigmoid(logits)                   ∈ [0, 1]
  mask   = proba > threshold                 binary uint8
```

**Total parameters**: 7,924,097 (all trainable)

### 3.3 Input Normalisation

Both images are normalised jointly per channel using the [p₂, p₉₈] percentile range:

```
joint  = concat(before_channel, after_channel)
lo, hi = percentile(joint, 2), percentile(joint, 98)
norm   = clip((pixel - lo) / (hi - lo), 0, 1)
```

Joint normalisation is critical: using separate statistics per image would destroy the radiometric difference signal that the model uses to detect change. This function is defined once in `preprocessor.py` and used identically in both training (OSCD dataset) and inference (Earth Engine pipeline).

### 3.4 Patch-Based Inference and Hanning Blending

For images larger than the training patch size, inference runs on a sliding window grid with 50% overlap:

```python
step     = patch_size × (1 - overlap)       # = patch_size / 2
pad_h    = (step - (H - patch_size) % step) % step
accum    = zeros(H + pad_h, W + pad_w, 1)
weights  = zeros(H + pad_h, W + pad_w)
win_2d   = outer(hanning(patch_size+2)[1:-1],  # strictly positive Hanning window
                  hanning(patch_size+2)[1:-1])

for each patch at (r, c):
    proba = infer(patch_before, patch_after)    # (ps, ps)
    accum [r:r+ps, c:c+ps] += proba × win_2d
    weights[r:r+ps, c:c+ps] += win_2d

result = accum / weights                        # weighted average → no seams
mask   = result[:H, :W] > threshold
```

The Hanning window is computed as `np.hanning(n+2)[1:-1]` to avoid zero-weight pixels at image borders (standard `np.hanning(n)` is 0 at index 0 and n-1).

### 3.5 Test-Time Augmentation

When `tta_enabled=True` (default), inference averages predictions over 4 flip variants:

```
TTA = mean(predict(b, a),
           predict(flip_h(b), flip_h(a))  → unflipped before storing,
           predict(flip_v(b), flip_v(a))  → unflipped before storing,
           predict(flip_hv(b), flip_hv(a)) → unflipped before storing)
```

TTA is applied per-patch before Hanning blending. Empirically adds ~1–2% F1 on LEVIR-CD at the cost of 4× per-patch compute.

---

## 4. Training on LEVIR-CD

### 4.1 Dataset

**LEVIR-CD** (Building Change Detection Dataset) is a large-scale binary change detection dataset:
- **Resolution**: 0.5 m/px (very high resolution aerial imagery)
- **Image size**: 1024 × 1024 px per pair
- **Split**: 447 train / 64 val / 128 test pairs
- **Change class**: building construction, demolition, and modification
- **Class imbalance**: ~2.5% change pixels (severe imbalance)
- **Channels**: RGB (3) → extended to 4 by appending synthetic NIR = mean(R, G, B)

The synthetic NIR channel is a pragmatic workaround: the model architecture expects 4-channel input to match the Sentinel-2 BGRNIR scheme, but LEVIR-CD provides only RGB. The synthetic channel preserves the channel count contract while containing no additional information.

### 4.2 Data Loader

`model/dataset.py` implements `LEVIRDataset`:

```python
# Each image loaded as (H, W, 4) float32 ∈ [0, 1]
img  = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
nir  = img.mean(axis=-1, keepdims=True)
img4 = np.concatenate([img, nir], axis=-1)

# Labels: grayscale PNG, value 255 = change → binary float32
lbl = (np.array(Image.open(path).convert("L")) > 128).astype(np.float32)
```

Augmentation (training only): random horizontal flip, vertical flip, and 90° rotation — all applied jointly to before/after/label to preserve spatial correspondence.

**Performance note on Windows Docker**: reading thousands of small PNGs from a volume-mounted Windows filesystem incurs ~2.5 s/batch due to NTFS→ext4 translation overhead. The dataset's `preload=True` mode loads all images into RAM once at `__init__` time (~3.1 GB for 445 train pairs) and serves subsequent crops from memory, reducing per-batch I/O to zero.

### 4.3 Loss Function

Class imbalance (97.5% background) is addressed with a composite loss:

```
L = 0.5 × L_wBCE + 0.5 × L_Dice

L_wBCE = BCE(logits, targets, pos_weight=10)
       # change pixels penalised 10× relative to background

L_Dice = 1 - (2 × Σ(p × t) + ε) / (Σp + Σt + ε)
       # ε = 1e-6; class-frequency invariant
```

Dice loss ensures the gradient does not collapse to the trivial "predict all background" solution even when BCE loss is dominated by the majority class.

### 4.4 Optimisation

```
Optimiser : AdamW   (lr=6e-5,  weight_decay=1e-4)
Schedule  : linear warmup (5 epochs) → cosine decay to 1% of peak LR
            λ(t) = 0.01 + 0.99 × 0.5 × (1 + cos(π × t))   t ∈ [0,1]
Gradient clipping : max_norm=1.0
Batch size : 16 (256×256 patches)
```

### 4.5 Training Phases on Kaggle (T4 × 2 GPU)

Training was conducted in multiple fine-tuning sessions on Kaggle's free GPU tier, using `model/kaggle_finetune.py`:

| Phase | Epochs | Peak LR | pos_weight | Notes |
|---|---|---|---|---|
| 1 | 1–50 | 6e-5 | 10.0 | Initial training from ImageNet init |
| 2 | 51–200 | 1e-5 | 5.0 | Continued, LR reduced |
| 3 | 201–325 | 5e-6 | 2.5 | Fine-tuning tail, tightening regularisation |

Checkpoint format:
```python
{
    "model":    state_dict,       # model weights
    "epoch":    int,              # epoch at which best val F1 was achieved
    "best_f1":  float,            # best val F1 score
    "args":     dict,             # training arguments for reproducibility
}
```

### 4.6 LEVIR-CD Results

| Metric | Value |
|---|---|
| F1 (change class) | **0.8605** |
| IoU (change class) | 0.7556 |
| Precision | 0.8812 |
| Recall | 0.8408 |
| Threshold | 0.50 |

Evaluation via `model/evaluate.py --split test` using `predict_large` (full 1024×1024 images, 50% overlap, TTA disabled for comparability).

**Checkpoint**: `gs://palimpsest-bucket/checkpoints/changeformer_levir.pth` (30.7 MB)

---

## 5. Fine-Tuning on OSCD (Sentinel-2)

### 5.1 The Resolution Mismatch Problem

The LEVIR-CD trained model was initially deployed directly against Sentinel-2 imagery. Production testing revealed a critical failure mode:

```
max_proba = 0.0009    (expected: > 0.30 for real change)
```

Root cause: **20× resolution gap**. LEVIR-CD images are 0.5 m/px (aerial), while Sentinel-2 is 10 m/px. At inference time, a single Sentinel-2 pixel corresponds to a 20×20 m ground patch — an area that would span 1,600 LEVIR-CD pixels. The model had never seen spatial patterns at this scale and produced near-zero probabilities regardless of actual ground change.

The solution is domain adaptation via fine-tuning on **OSCD** — a dataset of actual Sentinel-2 imagery with ground-truth change masks.

### 5.2 OSCD Dataset

**Onera Satellite Change Detection (OSCD)** dataset:
- **Source**: Kaggle (`soumikrakshit/onera-satellite-change-detection-dataset`)
- **Imagery**: Sentinel-2 Level-2A (surface reflectance), 10 m/px
- **Bands**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
- **Coverage**: 24 city pairs worldwide (14 labeled training cities + 10 test cities without public GT)
- **Change types**: urban expansion, construction, demolition, flood, vegetation loss
- **Change fraction**: 2%–30% per city (mean ≈ 10–15%)

**Kaggle-specific data conventions** discovered during data loading:

| Issue | Discovery | Fix |
|---|---|---|
| Path nesting | Cities at `oscd_raw/images/Onera Satellite Change Detection dataset - Images/{city}/` — one intermediate directory level | `_dig_if_single()`: if a directory has exactly 1 subdirectory, descend into it |
| Band file naming | Files stored as `S2A_OPER_MSI_L1C_TL_MTI__20160120T104345_A003020_T39QZG_B02.tif`, not `B02.tif` | `_load_bands()`: falls back to `glob("*_B02.tif")` when exact name not found |
| Mask encoding | Kaggle re-packaging uses `{1=no-change, 2=change}` instead of standard `{0=no-change, 1=change}` | Auto-detect: try `cm>1`, `cm>0`, `cm==0`, `cm>127` in order; use first giving 0.3–85% change fraction |
| Degenerate cities | Some cities (e.g., Paris) have placeholder masks with 100% change | Filter: reject cities outside 0.3%–85% change fraction during dataset scan |

### 5.3 Valid City Selection

After mask diagnostic scan, **13 cities** were retained from the 24 available:

| Split | Cities |
|---|---|
| Train (10) | abudhabi, aguasclaras, beihai, beirut, bercy, cupertino, hongkong, pisa, rennes, saclay_e |
| Val (3) | nantes, mumbai, bordeaux |

Selection rationale: nantes, mumbai, and bordeaux are geographically and climatically diverse, provide robust validation signal, and are among the best-labelled cities in OSCD.

### 5.4 OSCDDataset Implementation

`model/oscd_dataset.py` implements `OSCDDataset` with:

- **RAM caching**: all city image pairs are loaded and normalised once on first access, then served from `self._cache`. Total cache size ≈ 77 MB for 13 city pairs (float32, ~5.5 MB/pair).
- **Random crop**: each `__getitem__` draws a random 256×256 crop from a randomly selected city.
- **Augmentation**: random horizontal flip, vertical flip, random 90° rotation (training only).
- **Padding**: cities smaller than `patch_size` are reflected-padded before cropping (edge case handling).

```python
patches_per_city = 60    # training → 10 cities × 60 = 600 samples/epoch
patches_per_city = 30    # validation → 3 cities × 30 = 90 samples/epoch
```

### 5.5 Fine-Tuning Configuration

Fine-tuning was performed on Google Colab A100-SXM4-40GB via `model/colab_oscd.ipynb`:

```
Warm start     : LEVIR-CD checkpoint (epoch 325, F1=0.8605)
Epochs         : 150
Peak LR        : 5e-6  (10× lower than LEVIR training — pretrained feature preservation)
Warmup         : 5 epochs (linear)
LR schedule    : cosine decay to 1% of peak
pos_weight     : 5.0   (change class 5× — OSCD has higher change fraction than LEVIR)
Batch size     : 8
Patch size     : 256 px
Loss           : 0.5 × wBCE + 0.5 × Dice (same as LEVIR training)
Gradient clip  : max_norm=1.0
Optimiser      : AdamW, weight_decay=1e-4
Val every      : 10 epochs
```

Training dynamics (selected epochs):

| Epoch | Loss | F1 | IoU | Prec | Recall | LR |
|---|---|---|---|---|---|---|
| 10 | 0.4945 | 0.2521 | 0.1442 | 0.2624 | 0.2426 | 4.99e-06 |
| 20 | 0.4207 | 0.2692 | 0.1555 | 0.2424 | 0.3027 | 4.87e-06 |
| 120 | — | best ckpt | — | — | — | — |

### 5.6 Threshold Calibration

After training, the optimal decision threshold was calibrated on the 3-city val set (nantes/mumbai/bordeaux) via exhaustive sweep:

```
Threshold sweep: 0.03 → 0.54, step 0.03
Metric: macro F1 on change class
```

| Threshold | F1 | IoU | Precision | Recall |
|---|---|---|---|---|
| 0.03 | 0.2359 | 0.1337 | 0.1461 | 0.6130 |
| 0.18 | 0.2915 | 0.1706 | 0.2296 | 0.3989 |
| **0.24** | **0.2932** | **0.1718** | **0.2458** | **0.3632** |
| 0.27 | 0.2930 | 0.1716 | 0.2531 | 0.3478 |
| 0.54 | 0.2715 | 0.1570 | 0.3040 | 0.2452 |

Production deployment uses `confidence_threshold = 0.18` (slightly below the F1-optimal 0.24) to improve recall at the cost of marginally lower precision — appropriate for a change detection system where false negatives (missed changes) are more costly than false positives.

**Critical diagnostic improvement** — max probability on Sentinel-2 after OSCD fine-tuning:

| Model | max_proba on Sentinel-2 | Root cause |
|---|---|---|
| LEVIR-CD only | 0.0009 | 20× resolution mismatch — model blind to 10 m/px patterns |
| + OSCD fine-tuning | **0.9980** | Domain-adapted — model trained on actual Sentinel-2 |

**Checkpoint**: `gs://palimpsest-bucket/checkpoints/changeformer_oscd.pth` (30.7 MB)

---

## 6. Preprocessing Pipeline

`app/services/preprocessor.py` implements all preprocessing. Functions operate on `(H, W, C)` numpy arrays (channels-last convention throughout).

### 6.1 Joint Percentile Normalisation

```python
def normalize_pair(before, after, percentile_low=2.0, percentile_high=98.0):
    for c in range(n_channels):
        joint  = concat(before[...,c].ravel(), after[...,c].ravel())
        lo, hi = percentile(joint, 2), percentile(joint, 98)
        denom  = hi - lo
        if denom < 1e-8:
            out[...,c] = 0.0         # flat channel → zero to avoid NaN
        else:
            out[...,c] = clip((ch - lo) / denom, 0, 1)
```

The `joint` pool ensures both images share the same radiometric reference frame. Per-image normalisation would absorb true radiometric differences (the change signal) into the normalisation statistics.

### 6.2 Cloud Masking

```python
_CLOUD_MASK_CLASSES = {3, 8, 9, 10, 11}  # SCL classes
# 3=shadow, 8=cloud_med, 9=cloud_high, 10=thin_cirrus, 11=snow_ice

def apply_cloud_mask(image, scl):
    bad = np.isin(scl, list(_CLOUD_MASK_CLASSES))
    result = image.copy()
    result[bad] = 0.0
    return result
```

SCL (Scene Classification Layer, band B05 in export) is exported alongside spectral bands and discarded after masking.

### 6.3 Patch Extraction

```python
def _compute_padding(size, patch_size, step):
    if size < patch_size:
        return patch_size - size          # promote to at least one patch
    remainder = (size - patch_size) % step
    return (step - remainder) % step

def extract_patches(image, patch_size, overlap):
    step  = max(1, int(patch_size × (1 - overlap)))
    pad_h = _compute_padding(H, patch_size, step)
    image = np.pad(image, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
    return [image[r:r+ps, c:c+ps] for r,c in positions]
```

Reflect padding ensures border regions are not artificially zero-padded, which would bias the model's spatial attention near image boundaries.

### 6.4 Patch Reconstruction with Hanning Blending

```python
def reconstruct_from_patches(patches, shape, overlap):
    win_1d = np.hanning(patch_size + 2)[1:-1]     # strictly > 0 at all indices
    win_2d = np.outer(win_1d, win_1d)
    for (r, c), patch in zip(positions, patches):
        accum  [r:r+ps, c:c+ps] += patch × win_2d
        weights[r:r+ps, c:c+ps] += win_2d
    return (accum / weights)[:H, :W]
```

The Hanning window tapers each patch contribution toward its edges, so the final reconstruction is a smooth weighted average rather than a hard-boundary mosaic. `np.hanning(n+2)[1:-1]` ensures the window is strictly positive at every index, avoiding division-by-zero at image corners.

---

## 7. Earth Engine Retrieval Service

`app/services/earth_engine.py` provides a single public function `get_sentinel2_pair()`.

### 7.1 Scene Selection

```python
col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
       .filterBounds(ee.Geometry.Rectangle(bbox))
       .filterDate(start, end)                        # ±search_window_days
       .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
       .sort("CLOUDY_PIXEL_PERCENTAGE"))              # least-cloudy first
```

The `S2_SR_HARMONIZED` collection provides atmospherically corrected surface reflectance with harmonised processing baseline. Only scenes covering the full requested bounding box are considered (`.filterBounds`).

### 7.2 Export Pipeline

```python
task = ee.batch.Export.image.toCloudStorage(
    image          = img.select(["B2","B3","B4","B8","SCL"]).toUint16(),
    bucket         = GCS_BUCKET,
    fileNamePrefix = f"raw/before_{run_id}",
    region         = bbox_geom,
    scale          = 10,            # native 10 m/px
    fileFormat     = "GeoTIFF",
    formatOptions  = {"cloudOptimized": True},
    maxPixels      = int(1e10),
)
```

All bands are cast to `uint16` before export. Without the cast, mixing `uint16` spectral bands with the `uint8` SCL band causes EE to refuse the export task with a type-mismatch error.

### 7.3 Async Task Polling

Earth Engine exports are asynchronous. Polling uses exponential back-off:

```python
interval = 5          # initial poll interval (seconds)
deadline = now + 600  # 10-minute timeout

while now < deadline:
    sleep(interval)
    state = task.status()["state"]
    if state == "COMPLETED": return gcs_uri
    if state in ("FAILED", "CANCELLED"): raise RuntimeError(...)
    interval = min(interval × 2, 60)   # back-off ceiling: 60 s
```

### 7.4 EE Initialisation in Cloud Run

Cloud Run does not have a gcloud binary or an interactive auth flow. EE is initialised using `google.auth.default()` with explicit Earth Engine scopes:

```python
credentials, _ = google.auth.default(scopes=[
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
])
ee.Initialize(credentials=credentials, project=EE_PROJECT)
```

The Cloud Run service account `palimpsest-runner` has the `roles/earthengine.writer` role granted at the GCP project level.

---

## 8. Postprocessing and Vectorisation

`app/services/postprocessor.py`

### 8.1 Morphological Cleanup

```python
def clean_mask(mask, min_area_px=5):
    struct = ndi.generate_binary_structure(2, 1)    # 4-connectivity cross kernel
    closed = ndi.binary_closing(mask, structure=struct, iterations=2)
    labeled, _  = ndi.label(closed)
    sizes        = np.bincount(labeled.ravel())
    keep         = sizes >= min_area_px
    keep[0]      = False                            # never keep background label (0)
    return keep[labeled].astype(np.uint8)
```

At 10 m/px resolution, `min_area_px=5` corresponds to a minimum detectable change area of 500 m² (0.05 ha). The initial default of 50 pixels (5,000 m²) was too aggressive and suppressed real small-scale changes; reduced to 5 pixels after real-world testing.

Binary closing (2 iterations, 4-connectivity) fills single-pixel gaps and connects adjacent changed regions separated by at most 2 pixels (20 m at 10 m/px resolution).

### 8.2 Vectorisation

```python
def mask_to_polygons(mask, transform):
    for geom_dict, value in rasterio.features.shapes(mask, mask=(mask>0), transform=transform):
        poly = shapely.geometry.shape(geom_dict)
        features.append({"type": "Feature",
                         "geometry": geom_dict,
                         "properties": {"area_m2": round(poly.area, 2)}})
```

`rasterio.features.shapes()` traces connected regions using the affine transform from the source GeoTIFF, producing polygon coordinates in the image's native CRS (typically a UTM zone — a metric coordinate system where polygon `.area` is in m²).

### 8.3 Geometry Reprojection for BigQuery

A critical bug discovered in production: BigQuery GEOGRAPHY columns require WGS-84 (EPSG:4326, longitude/latitude in degrees). The vectorised polygons were in UTM (metres, e.g., Easting=482850, Northing=4523990) — BigQuery rejected these with:

```
Latitude must be between -90 and 90 degrees. Actual value was 4523990
```

Fix in `detect.py`:

```python
src_crs = pyproj.CRS(before_crs.to_epsg())
wgs84   = pyproj.CRS("EPSG:4326")
if src_crs != wgs84:
    project = pyproj.Transformer.from_crs(src_crs, wgs84, always_xy=True).transform
    union   = shapely.ops.transform(project, union)
wkt = union.wkt    # now in lon/lat degrees → accepted by BigQuery GEOGRAPHY
```

A secondary BigQuery polygon orientation issue — "Multipolygon contains polygons with overlap area larger than hemisphere" — occurs when the polygon exterior ring is wound counter-clockwise relative to what BigQuery expects. Fix: ensure `shapely.ops.unary_union` output is used directly (shapely enforces correct ring orientation automatically via its geometry normalisation).

---

## 9. REST API

FastAPI application served by Uvicorn at port 8080. All configuration via environment variables (Pydantic Settings).

### 9.1 Endpoints

#### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "changeformer-effb2-oscd-v1"
}
```

#### `POST /api/v1/detect`

**Request**:
```json
{
  "bbox":          [lon_min, lat_min, lon_max, lat_max],
  "date_before":   "2020-07-01",
  "date_after":    "2020-09-15",
  "max_cloud_pct": 20.0,
  "area_name":     "Beirut Port"
}
```

**Response**:
```json
{
  "detection_id":       "7ecf2771e2044a29b9ac1a0902313f28",
  "date_before_actual": "2020-07-10",
  "date_after_actual":  "2020-09-12",
  "change_area_m2":     145000.0,
  "change_pct":         8.73,
  "polygon_count":      12,
  "polygons":           [...],          // GeoJSON Feature array (WGS-84)
  "gcs_uri_before":     "gs://palimpsest-bucket/raw/before_a1b2c3d4.tif",
  "gcs_uri_after":      "gs://palimpsest-bucket/raw/after_a1b2c3d4.tif",
  "gcs_uri_mask":       "gs://palimpsest-bucket/results/7ecf2771_mask.tif",
  "model_version":      "changeformer-effb2-oscd-v1",
  "processing_time_s":  87.4
}
```

**Pipeline** (inside `detect.py`):
1. Validate request (bbox size, date ordering, cloud % range)
2. `get_sentinel2_pair()` → EE export → GCS URIs
3. `load_geotiff_from_gcs()` → `(H, W, 5)` arrays (4 spectral + SCL)
4. `normalize_pair()` → joint percentile normalisation
5. `apply_cloud_mask()` → zero-out SCL-masked pixels
6. `predict_large_debug()` → binary mask + max_proba diagnostic
7. `clean_mask()` → morphological cleanup
8. `mask_to_polygons()` → GeoJSON features in image CRS
9. Reproject polygons → WGS-84 via pyproj
10. `upload_mask_to_gcs()` → GeoTIFF with original CRS/transform
11. `save_detection()` → BigQuery insert
12. Return `DetectResponse`

#### `GET /api/v1/detect/{detection_id}`

Retrieves a previously stored detection from BigQuery by UUID.

#### `GET /docs`

Swagger UI (auto-generated by FastAPI).

### 9.2 Model Loading Strategy

The `ChangeDetector` instance is loaded once at application startup via FastAPI's `lifespan` context manager and stored as `app.state.detector`. This avoids the ~3-second model load penalty on every request.

Checkpoint loading supports both raw state_dict format and the wrapped `{"model": state_dict, "epoch": ..., "best_f1": ...}` format produced by the training loop.

---

## 10. AI Analysis Agent

The change detector answers *where* and *how much* changed. The analysis agent answers *what* and *why*. After a detection is stored, the client can request a natural-language interpretation that grounds the raw geometry in real-world events.

`app/services/analyst.py` implements `ChangeAnalyst`, a stateless per-request agent built on **Google Gemini** via the REST API.

### 10.1 Why REST instead of an SDK

The first implementation used a gRPC-based generative-AI SDK. Inside Cloud Run this consistently failed with `503 Illegal metadata`: the SDK's gRPC transport collides with Cloud Run's internal metadata server. The fix was to drop the SDK entirely and call the Gemini `generateContent` endpoint directly with `httpx` (already a project dependency) — no gRPC, no metadata conflict, no extra package.

```python
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
url      = f"{BASE_URL}/{settings.gemini_model}:generateContent"
resp     = httpx.post(url, params={"key": api_key}, json=body, timeout=90.0)
```

The model name is read from `settings.gemini_model` (env var `GEMINI_MODEL`), so the deployed model can be swapped with a single `gcloud run services update --update-env-vars` — no rebuild required.

### 10.2 Tools (function calling)

The agent is given two tools, declared in Gemini's `functionDeclarations` schema. Both are external, key-free HTTP services:

| Tool | Backend | Purpose |
|---|---|---|
| `lookup_location(lat, lon)` | OpenStreetMap Nominatim | Reverse-geocode the change centroid to a human-readable place (country, city, district, landmark) |
| `search_context(query)` | Wikipedia API | Retrieve background on events, projects, disasters, or developments that could explain the change |

### 10.3 Agentic Loop + Structured Output

```
1. Build a detection summary (centroid, dates, changed area in ha, polygon count).
2. Agentic loop (≤ 3 turns):
     model proposes functionCall(s) → execute tool(s) → feed functionResponse back
     repeat until the model stops calling tools or the turn budget is hit.
3. Structured generation: a final call with
     responseMimeType = "application/json", temperature = 0.2
   forces a strict JSON object:
     { location, summary, analysis, likely_causes[], confidence }
```

Splitting tool use (loop) from structured output (final JSON call) keeps the schema clean: the model reasons freely while it researches, then emits a single deterministic record.

### 10.4 Resilience

Gemini's free tier is rate-limited. `_gemini()` retries automatically on `429`/`503` with escalating back-off (10 → 20 → 30 → 45 → 60 s) and honours a `Retry-After` header when present. API keys are stripped from any error surfaced to the client (`key=...` → `key=***`).

### 10.5 Endpoint

#### `POST /api/v1/detect/{detection_id}/analyse`

Retrieves the stored detection from BigQuery and runs the agent. Returns `404` if the detection ID is unknown, `503` if no `GEMINI_API_KEY` is configured.

**Response**:
```json
{
  "detection_id":  "7ecf2771e2044a29b9ac1a0902313f28",
  "location":      "Port of Beirut, Marfaa, Beirut, Lebanon",
  "summary":       "A 174.66 ha change footprint at the Port of Beirut between Jul and Sep 2020.",
  "analysis":      "The change area is consistent with the catastrophic 4 August 2020 ammonium nitrate explosion that destroyed the port's grain silos and surrounding warehouses...",
  "likely_causes": ["Ammonium nitrate explosion (Aug 2020)", "Demolition of damaged structures", "Post-blast debris clearance"],
  "confidence":    "HIGH",
  "context_used":  ["[lookup_location(...)] → Port of Beirut...", "[search_context(...)] → [Wikipedia: 2020 Beirut explosion] ..."],
  "generated_at":  "2026-06-02T20:15:03+00:00"
}
```

The `context_used` array exposes every tool call and its result for transparency — the user can audit exactly what external evidence the interpretation was built on.

The API key lives in Secret Manager and is mounted into Cloud Run via `--set-secrets=GEMINI_API_KEY=gemini-api-key:latest`; it is never present in source or images.

---

## 11. GCP Infrastructure

### 10.1 Resources

| Resource | Name | Purpose |
|---|---|---|
| Project | `project-8c7ca821-aa7a-45ea-88b` | GCP project |
| Cloud Storage | `palimpsest-bucket` | Raw imagery, checkpoints, mask results |
| BigQuery dataset | `palimpsest_results` | Change detection records |
| BigQuery table | `change_detections` | Per-detection rows with GEOGRAPHY |
| Artifact Registry | `palimpsest-repo` | Docker images |
| Cloud Run service | `palimpsest-api` | Production API |
| Service account | `palimpsest-runner` | Minimum-privilege execution identity |

### 10.2 BigQuery Schema

```json
[
  {"name": "detection_id",     "type": "STRING",    "mode": "REQUIRED"},
  {"name": "area_name",        "type": "STRING",    "mode": "NULLABLE"},
  {"name": "date_before",      "type": "DATE",      "mode": "REQUIRED"},
  {"name": "date_after",       "type": "DATE",      "mode": "REQUIRED"},
  {"name": "change_area_m2",   "type": "FLOAT64",   "mode": "NULLABLE"},
  {"name": "change_pct",       "type": "FLOAT64",   "mode": "NULLABLE"},
  {"name": "geometry",         "type": "GEOGRAPHY", "mode": "NULLABLE"},
  {"name": "confidence_score", "type": "FLOAT64",   "mode": "NULLABLE"},
  {"name": "gcs_uri_before",   "type": "STRING",    "mode": "NULLABLE"},
  {"name": "gcs_uri_after",    "type": "STRING",    "mode": "NULLABLE"},
  {"name": "gcs_uri_mask",     "type": "STRING",    "mode": "NULLABLE"},
  {"name": "model_version",    "type": "STRING",    "mode": "NULLABLE"},
  {"name": "processed_at",     "type": "TIMESTAMP", "mode": "REQUIRED"}
]
```

The `geometry` column uses BigQuery's native `GEOGRAPHY` type, enabling geospatial queries such as:

```sql
SELECT detection_id, change_area_m2, geometry
FROM `project.palimpsest_results.change_detections`
WHERE ST_Intersects(geometry, ST_GeogFromText('POLYGON((...))'))
  AND date_after >= '2020-01-01'
```

### 10.3 GCS Lifecycle Policy

Raw imagery is auto-deleted after 90 days via a lifecycle rule on the `raw/` prefix (`infra/gcs_lifecycle.json`). Mask GeoTIFFs in `results/` and checkpoints in `checkpoints/` have no lifecycle rule and are retained indefinitely.

### 10.4 Cloud Run Configuration

```yaml
memory:        4 GiB      # model weights (30 MB) + image buffers
cpu:           2 vCPU
timeout:       3600s       # 1 hour — EE export can take 2–3 min per image
concurrency:   4           # up to 4 simultaneous detections per instance
min-instances: 0           # scale to zero when idle (cost optimisation)
max-instances: 3           # cap on parallel instances
```

---

## 12. CI/CD Pipeline

`infra/cloudbuild.yaml` defines a 3-step Cloud Build pipeline:

```
Step 0 — Download checkpoint from GCS
  gcloud storage cp gs://palimpsest-bucket/checkpoints/changeformer_oscd.pth
                    checkpoints/changeformer_oscd.pth
  # Checkpoint is excluded from the build context (large binary)
  # Downloaded here so it gets picked up by COPY . . in the Dockerfile

Step 1 — Build Docker image
  docker build -t .../{image}:{BUILD_ID} -t .../{image}:latest .

Step 2 — Push to Artifact Registry
  docker push --all-tags .../palimpsest-repo/palimpsest-api

Step 3 — Deploy to Cloud Run
  gcloud run deploy palimpsest-api
    --image=...:{BUILD_ID}    # pinned to exact build, not :latest
    --region=us-central1
    --memory=4Gi --cpu=2
    --timeout=3600
    --concurrency=4
    --min-instances=0 --max-instances=3
    --allow-unauthenticated
```

**Deployment history** (progressive model improvement):

| Revision | Model checkpoint | KEY change |
|---|---|---|
| 00001–00004 | `changeformer_levir.pth` | Initial LEVIR-CD deployment |
| 00005 | `changeformer_levir.pth` | Geometry reprojection fix (UTM→WGS-84) |
| 00007 | `changeformer_oscd.pth` | OSCD fine-tuned model (max_proba 0.0009→0.9980) |
| 00008 | `changeformer_oscd.pth` | MODEL_VERSION string updated |
| 00009 | `changeformer_oscd.pth` | Cloud Run timeout 300s→3600s; pyproj added; polygon orientation fix |
| 00010 | `changeformer_oscd.pth` | Demo UI: coordinate inputs, demo presets, area size guard |
| 00020+ | `changeformer_oscd.pth` | AI analysis agent (Gemini REST); `GEMINI_API_KEY` via Secret Manager |
| 00033+ | `changeformer_oscd.pth` | Map viewport fix (invalidateSize); 429 retry/back-off; key masking in errors |

---

## 13. Demo Interface

`demo/map/` — a zero-dependency Leaflet.js application served as static files by FastAPI at `/demo/`.

**URL**: `https://palimpsest-api-524824658539.us-central1.run.app/demo/`

### Features

- **Basemap toggle**: OpenStreetMap / Esri World Imagery satellite
- **Bounding box draw**: `Shift+drag` on map
- **Coordinate input**: W/E/S/N fields with "Apply coordinates →" button
- **Demo location presets**: 10 pre-configured locations with calibrated bboxes (≤ 8 km × 8 km), dates, cloud tolerance, and area labels
- **Area size guard**: rejects bboxes > 15 km per side before dispatching the request
- **Progress bar**: Hanning-weighted bar with phase-labelled status text and elapsed timer
- **Results card**: scene dates, changed area, change %, polygon count, processing time, detection ID (click to copy)
- **Change overlay**: GeoJSON polygons rendered in red (opacity 0.45) over the satellite basemap
- **Analyse with AI**: one-click button that calls the Gemini agent and renders the interpretation card — location, summary, scientific analysis, likely causes, and a colour-coded confidence badge (HIGH / MEDIUM / LOW)

### Demo Presets

| Location | Dates | Change type |
|---|---|---|
| 🇱🇧 Beirut — port explosion | Jul→Sep 2020 | Catastrophic industrial destruction (~3 km²) |
| 🇦🇪 Dubai — Creek Harbour | 2017→2022 | Coastal construction, land reclamation |
| 🇸🇦 NEOM — The Line | 2021→2024 | Desert mega-project earthworks |
| 🇸🇦 Riyadh — northern expansion | 2017→2023 | Suburban grid construction |
| 🇨🇳 Shenzhen — Qianhai Bay | 2016→2022 | Bay reclamation and urban infill |
| 🇺🇸 Las Vegas — Henderson | 2016→2023 | Suburban residential expansion |
| 🇧🇷 Amazon — Pará | 2018→2022 | Deforestation frontier |
| 🇺🇦 Kakhovka — flood zone | Apr→Aug 2023 | Nova Kakhovka dam breach inundation |
| 🇪🇬 New Alamein | 2018→2023 | Desert city construction ex nihilo |
| 🇸🇦 NEOM — Sindalah island | 2022→2024 | Offshore island artificial construction |

---

## 14. Performance Summary

### 14.1 Model Performance

| Benchmark | F1 | IoU | Notes |
|---|---|---|---|
| LEVIR-CD test set | **0.8605** | 0.7556 | Full images, predict_large, threshold=0.50 |
| OSCD val set (3 cities) | **0.2932** | 0.1718 | Patch-based, threshold=0.24 |

The lower OSCD F1 reflects the inherent difficulty of the dataset: small change fractions (1–5%), imprecise ground-truth masks at 10 m/px resolution, and only 3 val cities. The critical production-relevant diagnostic is `max_proba`: 0.9980 on a real Sentinel-2 scene confirms the model is genuinely responding to land-cover change at satellite resolution.

### 14.2 Inference Timing (CPU, Cloud Run)

| Area | Approx. size | EE export | Inference | Total |
|---|---|---|---|---|
| Beirut port (3×3 km) | 300×330 px | ~60s | ~5s | ~70s |
| Typical demo (5×5 km) | 500×500 px | ~90s | ~10s | ~100s |
| Large area (10×10 km) | 1000×1000 px | ~150s | ~45s | ~200s |

All inference runs on CPU (no GPU on Cloud Run). EE export dominates total latency; model inference is O(n_patches) where each 256×256 patch takes ~0.3s with TTA.

### 14.3 Known Limitations

- **Seasonal radiometric variation**: the model was fine-tuned on OSCD which spans multiple seasons; joint normalisation partially compensates, but large seasonal changes (leaf-on/leaf-off, snow cover) can generate false positives.
- **Cloud masking**: scenes with >20% cloud cover may contain residual cloud artefacts post-masking that the model may flag as change.
- **Spatial resolution floor**: at 10 m/px, changes smaller than ~500 m² (5 pixels, ~20×25 m footprint) are below the minimum detection threshold.
- **Val set size**: 3 OSCD cities is a small evaluation set; threshold calibration may not generalise perfectly to all geographic regions.

---

## 15. Configuration Reference

All configuration is injected via environment variables and managed by Pydantic Settings (`app/core/config.py`). No secrets are present in source code.

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT_ID` | required | GCP project ID |
| `GCP_REGION` | `us-central1` | Cloud Run deployment region |
| `GCS_BUCKET_NAME` | required | Cloud Storage bucket |
| `GCS_RAW_PREFIX` | `raw/` | Prefix for exported Sentinel-2 images |
| `GCS_RESULTS_PREFIX` | `results/` | Prefix for output mask GeoTIFFs |
| `GCS_CHECKPOINTS_PREFIX` | `checkpoints/` | Prefix for model checkpoints |
| `BQ_DATASET` | `palimpsest_results` | BigQuery dataset name |
| `BQ_TABLE` | `change_detections` | BigQuery table name |
| `EE_PROJECT` | required | Earth Engine project ID |
| `MODEL_CHECKPOINT_PATH` | `checkpoints/changeformer_oscd.pth` | Path to model weights (relative to /app) |
| `CONFIDENCE_THRESHOLD` | `0.18` | Sigmoid threshold for change/no-change |
| `PATCH_SIZE` | `512` | Sliding window patch size in pixels |
| `TTA_ENABLED` | `True` | 4-way flip test-time augmentation |
| `MIN_CHANGE_AREA_PX` | `5` | Minimum connected component size (pixels) |
| `GEMINI_API_KEY` | optional | Google AI Studio key for the analysis agent (Secret Manager) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used by the analysis agent |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

*Model checkpoint DOI / arXiv reference for the ChangeFormer architecture:*
> Bandara, W. G. C., & Patel, V. M. (2022). **ChangeFormer: A Transformer-Based Siamese Network for Change Detection**. IGARSS 2022. [arXiv:2201.01293](https://arxiv.org/abs/2201.01293)
