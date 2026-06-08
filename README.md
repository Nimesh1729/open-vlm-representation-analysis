# Open VLM Representation Analysis

This is a research-grade representation-analysis project for studying how open vision-language models encode, align, and organize image and text embeddings.

This project begins with CLIP as CLIP provides the cleanest entry point into multimodal representation learning:

```text
image → image embedding
text  → text embedding
        ↓
shared embedding space
```

My goal is not to train a large model, with due respect to my hardware constraints. My goal is to understand how pretrained vision-language models behave internally and how their embeddings can be analyzed using certain tools that I have previously used for other similar analysis projects.

---

## Project Goal

This repository tries to investigate the question:

```text
How do open vision-language models align image and text representations?
```

The current version of this repo focuses on CLIP image-text embeddings and asks:

* Can CLIP retrieve the correct caption for an image?
* Are matching image-text pairs more similar than non-matching pairs?
* Do image and text embeddings form meaningful clusters?
* Is there a visible modality gap between image and text embeddings?
* Which semantic classes are easiest or hardest for CLIP to separate?

The current dataset contains four classes:

* galaxy
* dog
* cat
* car

The small controlled dataset allows us to inspect representation behavior before moving to larger datasets, astronomy-specific data, SDSS/TESS subsets.

---

## Motivation

This project follows from a previous Transformer Representation Analysis project, where text-only transformer embeddings were studied using:

* PCA
* UMAP
* cosine similarity
* class separation
* layer-wise analysis
* CKA
* attention analysis

The next natural step I think is to move from:

```text
text embeddings
```

to:

```text
image + text embeddings
```

This repository applies the same representation-analysis mindset to CLIP-style multimodal models.

The broader roadmap is:

```text
Transformer Representations
        ↓
Open VLM / CLIP Representation Analysis
        ↓
Astronomy Data Analysis
        ↓
Reasoning Audit
```

---

## Current Model

The current model is:

```text
openai/clip-vit-base-patch32
```

CLIP consists of:

```text
image encoder
text encoder
projection into shared embedding space
```

For the current model, both image and text embeddings have shape:

```text
(1, 512)
```

for a single sample.

---

## Repository Structure
Thank you free chatgpt for generating me this cool repo structure.
```text
Open_VLM_Representation/
│
├── configs/
│   └── base.yaml
│
├── data/
│   └── samples/
│       ├── images/
│       │   ├── galaxy_1.jpg
│       │   ├── galaxy_2.jpg
│       │   ├── ...
│       │   ├── dog_1.jpg
│       │   ├── ...
│       │   ├── cat_1.jpg
│       │   ├── ...
│       │   └── car_1.jpg
│       │
│       └── captions.csv
│
├── scripts/
│   ├── compare_clip_texts.py
│   ├── extract_dataset_embeddings.py
│   ├── inspect_clip_embeddings.py
│   ├── run_class_separation_analysis.py
│   ├── run_cross_modal_analysis.py
│   ├── run_full_pipeline.py
│   ├── run_joint_visual_analysis.py
│   ├── run_retrieval_analysis.py
│   ├── run_similarity_analysis.py
│   └── run_visual_analysis.py
│
├── src/
│   ├── analysis/
│   │   ├── class_separation.py
│   │   ├── cross_modal.py
│   │   ├── pca.py
│   │   ├── retrieval.py
│   │   ├── similarity.py
│   │   └── umap.py
│   │
│   ├── extraction/
│   │   ├── clip_embeddings.py
│   │   ├── dataset.py
│   │   └── dataset_embeddings.py
│   │
│   ├── models/
│   │   └── load_model.py
│   │
│   ├── utils/
│   │   ├── cli.py
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── paths.py
│   │   └── reproducibility.py
│   │
│   └── visualization/
│       ├── joint_scatter.py
│       └── scatter.py
│
├── tests/
│   ├── test_config_loader.py
│   ├── test_dataset.py
│   ├── test_load_model.py
│   ├── test_paths.py
│   └── test_reproducibility.py
│
├── outputs/
├── logs/
├── README.md
├── research_journal.md
├── requirements.txt
└── pyproject.toml
```

---

## Setup

I created and activated a virtual environment.

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

To install the dependencies:

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

This allows imports such as:

```python
from src.utils.config_loader import load_config
```

to work cleanly across scripts and tests. Otherwise the pathing will just break.

---

## Important Dependency Note

This project uses Hugging Face `transformers`.

A recent `transformers` version caused compatibility issues with the installed PyTorch version because it attempted to access:

```text
torch.float8_e8m0fnu
```

To avoid this, the project pins:

```text
transformers==4.44.2
```

This version is stable for the current CLIP workflow. I have no idea why this happened.

The project also uses:

```text
torch
torchvision
transformers==4.44.2
pillow
pyyaml
types-PyYAML
numpy
pandas
matplotlib
scikit-learn
umap-learn
pytest
```

---

## Configuration

The main configuration file is:

```text
configs/base.yaml
```

Current structure:

```yaml
model:
  name: openai/clip-vit-base-patch32

system:
  seed: 42
  device: cuda

data:
  captions_csv: data/samples/captions.csv
  sample_image: data/samples/sample_image.jpg
  sample_text: "A galaxy with bright stars."
  candidate_texts:
    - "A galaxy with bright stars."
    - "A dog sitting in a park."
    - "A cat sleeping on a sofa."
    - "A car parked on a street."

logging:
  level: INFO
```

The project is config-driven so paths, models, devices, and data sources can be changed without modifying script internals.

---

## Dataset

The current dataset contains 20 image-caption pairs:

```text
5 galaxy images
5 dog images
5 cat images
5 car images
```

The CSV format is:

```csv
image_path,caption,label
data/samples/images/galaxy_1.jpg,A spiral galaxy with bright stars.,galaxy
data/samples/images/dog_1.jpg,A dog standing outdoors.,dog
data/samples/images/cat_1.jpg,A cat resting on bed.,cat
data/samples/images/car_1.jpg,A car parked somewhere.,car
```

The dataset is intentionally small and interpretable.

Its purpose is to verify and analyze the representation pipeline before moving toward:

* larger image-caption datasets
* astronomy-only image-caption datasets
* SDSS cutouts
* spectra-image-text alignment

---

## Running the Project

### 1. Inspect Single Image/Text Embeddings

```bash
python scripts/inspect_clip_embeddings.py --config configs/base.yaml
```

Expected output:

```text
Text embedding shape:  (1, 512)
Image embedding shape: (1, 512)
Cosine similarity: ...
```

This confirms that CLIP maps image and text inputs into the same embedding dimension.

---

### 2. Compare One Image Against Candidate Captions

```bash
python scripts/compare_clip_texts.py --config configs/base.yaml
```

This compares one image against multiple text captions.

Example result from the initial smoke test:

```text
Similarity 0.1703 | A galaxy with bright stars.
Similarity 0.1140 | A dog sitting in a park.
Similarity 0.0824 | A cat sleeping on a sofa.
Similarity 0.1481 | A car parked on a street.
```

The matching astronomy caption was ranked highest.

---

### 3. Extract Dataset Embeddings

```bash
python scripts/extract_dataset_embeddings.py --config configs/base.yaml
```

This produces (thank you chatgpt again for repository structure):

```text
outputs/clip/embeddings/
├── image_embeddings.npy
├── text_embeddings.npy
├── labels.npy
└── captions.npy
```

For the current 20-sample dataset:

```text
image_embeddings: (20, 512)
text_embeddings:  (20, 512)
labels:           (20,)
captions:         (20,)
```

---

### 4. Running Image-to-Text Retrieval

```bash
python scripts/run_retrieval_analysis.py --config configs/base.yaml
```

This answers:

```text
Given an image embedding, which caption embedding is most similar?
```

Outputs (Thanks for the third time chatgpt for the structure format):

```text
outputs/clip/analysis/retrieval/
├── image_to_text_similarity_matrix.npy
├── image_to_text_retrieval_results.csv
└── image_to_text_metrics.csv
```

Important note:

Exact-caption retrieval is strict. For example, a galaxy image may retrieve another galaxy caption instead of its exact paired caption. This should not necessarily be interpreted as a failure, because CLIP performs semantic retrieval rather than file-name matching.

---

### 5. Run Similarity Matrix Analysis

```bash
python scripts/run_similarity_analysis.py --config configs/base.yaml
```

This saves the full image-text similarity matrix:

```text
outputs/clip/analysis/similarity/
├── image_text_similarity_matrix.npy
└── image_text_similarity_matrix.csv
```

This matrix shows how every image relates to every caption.

---

### 6. Run Cross-Modal Alignment Analysis

```bash
python scripts/run_cross_modal_analysis.py --config configs/base.yaml
```

This computes:

```text
matching_mean
non_matching_mean
alignment_gap
```

For the 20-sample dataset:

| Metric            |  Value |
| ----------------- | -----: |
| Matching mean     | 0.2712 |
| Non-matching mean | 0.1856 |
| Alignment gap     | 0.0856 |

Interpretation:

```text
matching image-text pairs are more similar than non-matching pairs
```

However, with multiple examples per class, some non-matching pairs are still semantically related. For example:

```text
galaxy_1 image ↔ galaxy_4 caption
```

is technically non-matching but semantically valid.

---

### 7. Run Visual Analysis

```bash
python scripts/run_visual_analysis.py --config configs/base.yaml
```

This generates PCA and UMAP plots for image embeddings and text embeddings:

```text
outputs/clip/visualizations/
├── image_pca.png
├── image_umap.png
├── text_pca.png
└── text_umap.png
```

PCA explained variance for the 20-sample dataset:

| Embedding Type   |    PC1 |    PC2 |  Total |
| ---------------- | -----: | -----: | -----: |
| Image embeddings | 0.2798 | 0.2185 | 0.4982 |
| Text embeddings  | 0.2489 | 0.1745 | 0.4234 |

Interpretation:

Image and text embedding spaces both show meaningful low-dimensional structure without collapsing into a single dominant direction.

---

### 8. Run Joint Image-Text PCA

```bash
python scripts/run_joint_visual_analysis.py
```

This creates a joint PCA plot containing both image and text embeddings:

```text
outputs/clip/visualizations/joint/
└── joint_image_text_pca.png
```

Joint PCA explained variance:

| Component | Explained Variance |
| --------- | -----------------: |
| PC1       |             0.4063 |
| PC2       |             0.1197 |

Interpretation:

The joint PCA plot reveals a clear modality gap.

Images and texts share semantic structure, but they still occupy visibly separated regions in the low-dimensional projection.

A useful interpretation is:

```text
CLIP embedding space = shared semantic structure + modality offset
```

---

### 9. Run Class Separation Analysis

```bash
python scripts/run_class_separation_analysis.py
```

This computes:

```text
within-class similarity
between-class similarity
separation_score
```

where:

```text
separation_score = within_similarity - between_similarity
```

Current results:

| Label  | Within Similarity | Between Similarity | Separation Score |
| ------ | ----------------: | -----------------: | ---------------: |
| galaxy |            0.3038 |             0.1779 |           0.1260 |
| dog    |            0.2515 |             0.1663 |           0.0852 |
| car    |            0.2229 |             0.1407 |           0.0822 |
| cat    |            0.2595 |             0.1787 |           0.0808 |

Interpretation:

* Galaxy is the most strongly separated class.
* Cat is the weakest separated class.
* Cat and dog show partial overlap, which is expected for visually similar animal classes.
* CLIP separates galaxy images/captions especially well from everyday object classes.

---

## Key Findings

### 1. CLIP Produces Aligned Image and Text Embeddings

For a single image and caption:

```text
text embedding shape:  (1, 512)
image embedding shape: (1, 512)
```

Both modalities are projected into the same 512-dimensional embedding space.

---

### 2. CLIP Performs Semantic Retrieval, Not Exact Caption Matching

On the 20-sample dataset, many image queries retrieved a caption from the correct class but not the exact paired caption.

Example:

```text
galaxy_1 image
correct caption:   A spiral galaxy with bright stars.
predicted caption: An astronomical image of a galaxy.
```

This is semantically reasonable.

Therefore, exact-caption accuracy is too strict once there are multiple captions per class.

Class-level retrieval is more meaningful for this dataset.

---

### 3. Matching Pairs Are More Similar Than Non-Matching Pairs

For the 20-sample dataset:

```text
matching_mean     = 0.2712
non_matching_mean = 0.1856
alignment_gap     = 0.0856
```

This shows that CLIP places corresponding image-text pairs closer together than unrelated pairs.

---

### 4. A Modality Gap Exists

Joint PCA shows that the largest direction of variance separates:

```text
image embeddings
```

from:

```text
text embeddings
```

This means CLIP does not make image and text embeddings identical in distribution.

Instead, the evidence suggests:

```text
shared semantic structure + modality offset
```

---

### 5. Galaxy Is the Best-Separated Class

Galaxy achieved the highest class separation score:

```text
galaxy separation = 0.1260
```

This indicates that galaxy images and galaxy captions form the most distinct multimodal concept cluster in the current dataset.

This is consistent with the visual PCA/UMAP plots.

---

### 6. Cat Is the Hardest Class

Cat achieved the lowest separation score:

```text
cat separation = 0.0808
```

Cat embeddings partially overlap with dog embeddings, likely because both classes share visual and semantic features such as:

* pet animal
* fur
* face
* indoor/outdoor scenes
* camera-facing poses

---

### 7. Multiple Analyses Agree

The following analyses all tell a consistent story:

| Analysis              | Finding                                                        |
| --------------------- | -------------------------------------------------------------- |
| Retrieval             | Images usually retrieve semantically correct class captions    |
| Similarity matrix     | Same-class image-text similarities are high                    |
| Cross-modal alignment | Matching pairs exceed non-matching pairs                       |
| PCA/UMAP              | Classes form visible clusters                                  |
| Joint PCA             | Images and texts share semantic structure but remain separated |
| Class separation      | Galaxy is most distinct; cat is hardest                        |

This agreement across methods increases confidence in the result.

---

## Research Interpretation

The current working interpretation is:

```text
CLIP aligns image and text into a shared semantic embedding space,
but image and text embeddings still preserve a modality-dependent offset.

Semantic class structure is clearly present in both modalities.

Galaxy forms the most distinct concept cluster,
while cat/dog overlap is the strongest source of confusion.
```

This repository therefore demonstrates the basic representation-analysis toolkit needed for future multimodal astronomy work.

---

## Development Standards

This repository follows the same standards used across the broader research roadmap:

* Google Python Style Guide
* type hints where practical
* Ruff linting (not strict)
* mypy static type checking (not strict)
* pytest tests
* config-driven experiments
* reproducible random seeds
* centralized utilities
* professional repo structure
* README and research journal maintenance

Run tests:

```bash
pytest
```

Run Ruff:

```bash
ruff check .
```

Run mypy:

```bash
mypy src
```

---

## Current Tests

The repository includes tests for:

* config loading
* reproducibility
* path utilities
* dataset loading
* CLIP model-loader wrappers

The model-loader tests use mocks so they do not download CLIP weights during normal test runs.

---

## What I didn't Commit

The pictures I used for the data I did not commit them, although any user can choose their own images following the structure for naming:

```text
filename_iteration.jpg
```


## Current Status

This repository currently supports:

```text
✓ CLIP model loading
✓ CLIP processor loading
✓ single image/text embedding inspection
✓ image-caption dataset loading
✓ dataset-level image/text embedding extraction
✓ image-to-text retrieval analysis
✓ image-text similarity matrix analysis
✓ cross-modal alignment metrics
✓ image/text PCA visualization
✓ image/text UMAP visualization
✓ joint image-text PCA visualization
✓ paired image-text joint scatter plotting
✓ class separation analysis
✓ tests and static checks
```

This is a complete first-stage CLIP representation-analysis pipeline.

---


## Summary

This project demonstrates how to analyze CLIP-style multimodal representations using a research-grade pipeline.

The key result is:

```text
CLIP produces meaningful image-text alignment and class-level semantic structure,
while still preserving a visible modality gap between image and text embeddings.
```

This makes the repository a strong foundation for future open VLM analysis, astronomy-specific multimodal datasets, and representation alignment.
