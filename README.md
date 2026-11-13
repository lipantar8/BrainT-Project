# Glioma Segmentation from Multimodal 3D MRI

This repository contains the implementation developed for the **AI in Biomedicine** course project on automatic glioma segmentation from multimodal 3D MRI.

The project focuses on segmenting three tumor subregions from four MRI modalities:

- **FLAIR**
- **T1**
- **T1ce**
- **T2**

The main baseline is a **3D U-Net** trained using patient-level **5-fold cross-validation**.

---

## Project Overview

The goal of the project is to develop and evaluate deep learning models for volumetric glioma segmentation.

The dataset contains MRI scans from **1384 patients across 8 centers**. MRI modalities are co-registered and resampled to approximately **1 × 1 × 1 mm**.

The available segmentation labels are:

| Label | Region |
|---|---|
| 0 | Background |
| 1 | Necrotic core |
| 2 | FLAIR signal hyperintensity |
| 4 | Enhancing tumor |

For model training, the original labels are internally remapped to four consecutive classes:

| Original label | Model label | Region |
|---|---|---|
| 0 | 0 | Background |
| 1 | 1 | Necrotic core |
| 2 | 2 | FLAIR signal hyperintensity |
| 4 | 3 | Enhancing tumor |

---

## Repository Structure

```text
.
├── README.md
│
├── notebooks/
│   ├── 01_dataset_exploration_and_preprocessing.ipynb
│   └── 02_3d_unet_baseline.ipynb
│
└── results/
    └── ...
```

The exact filenames can be changed to match the notebooks uploaded to the repository.

---

## Notebook 1 — Dataset Exploration and Preprocessing

The first notebook is dedicated to dataset preparation and exploratory analysis.

It includes:

- dataset and metadata inspection;
- patient ID normalization;
- center/cohort analysis;
- recursive MRI and segmentation-mask discovery;
- MRI modality availability checks;
- segmentation label inspection;
- MRI and tumor visualization;
- patient-level train/validation/test split analysis;
- construction of the segmentation manifest;
- construction of the MAE manifest;
- creation of a fixed **5-fold cross-validation manifest**.

The final supervised dataset contains:

- **1003 development patients** used for 5-fold cross-validation;
- **177 independent test patients**, kept completely separate from model development.

The five folds are stored in:

```text
5fold_manifest.csv
```

This same manifest is intended to be reused across supervised experiments to ensure that different models are evaluated on the same patient-level splits.

---

## Notebook 2 — 3D U-Net Baseline

The second notebook implements the baseline segmentation model.

### Model architecture

The model is a custom **3D U-Net** designed for volumetric MRI segmentation.

```text
4 MRI modalities
       │
       ▼
   3D U-Net
       │
       ▼
4 segmentation classes
```

The architecture includes:

- 4 input channels;
- 3D convolutional encoder;
- bottleneck representation;
- 3D decoder;
- skip connections;
- 4 output channels corresponding to the segmentation classes.

### Configuration

| Parameter | Value |
|---|---:|
| Architecture | 3D U-Net |
| Input channels | 4 |
| Output classes | 4 |
| Initial filters | 8 |
| Patch size | 64 × 64 × 64 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Loss | 0.5 CE + 0.5 Soft Dice |
| Maximum epochs | 50 |
| Early stopping patience | 10 |
| Cross-validation | 5-fold |

---

## Preprocessing

MRI volumes are normalized independently using non-zero voxels.

The preprocessing includes:

1. removal of background intensity contribution;
2. percentile clipping using the 0.5th and 99.5th percentiles;
3. z-score normalization;
4. zero background preservation.

The four MRI modalities are stacked into a single 4-channel input tensor:

```text
[FLAIR, T1, T1ce, T2]
```

---

## Patch Sampling

Because the MRI volumes are 3D, training is performed using volumetric patches of:

```text
64 × 64 × 64
```

Tumor-centered sampling is used with higher probability when tumor voxels are available.

This increases the probability that training patches contain relevant tumor regions while still allowing random sampling.

---

## Data Augmentation

Spatial augmentations are applied consistently to both the MRI volume and segmentation mask.

The baseline uses:

- random flips along the three spatial axes;
- random 90° rotations;
- random MRI intensity scaling;
- random MRI intensity shifting.

Intensity transformations are applied only to the MRI channels and **not** to the segmentation mask.

The notebook includes visual checks comparing the original and augmented MRI/segmentation pairs.

---

## Loss Function

The training objective combines Cross Entropy and Soft Dice Loss:

```text
Loss = 0.5 × Cross Entropy + 0.5 × Soft Dice Loss
```

The Soft Dice component focuses on the foreground tumor classes, while Cross Entropy provides voxel-level classification supervision.

---

## Evaluation Metrics

The segmentation task is evaluated using:

- **Dice coefficient**
- **Intersection over Union (IoU)**
- **Hausdorff Distance (HD)**
- **95th percentile Hausdorff Distance (HD95)**

Metrics are reported separately for:

- necrotic core;
- FLAIR signal hyperintensity;
- enhancing tumor.

The background class is excluded from the foreground segmentation metrics.

---

## Cross-Validation Strategy

The development set contains **1003 patients**.

A fixed **5-fold patient-level cross-validation** strategy is used.

Each fold contains:

```text
Training patients
+
Validation patients
```

with no patient appearing in both sets for the same fold.

The independent test set of **177 patients** is not used during cross-validation or model selection.

This ensures that the final test evaluation remains independent from the development process.

---

## Training

Each fold starts from a **new 3D U-Net initialized from scratch**.

Training uses:

- AdamW;
- mixed-precision training;
- batch size 2;
- early stopping;
- best-checkpoint saving.

The best checkpoint for each fold is saved separately.

Example:

```text
checkpoints/
├── unet3d_fold0_best.pth
├── unet3d_fold1_best.pth
├── unet3d_fold2_best.pth
├── unet3d_fold3_best.pth
└── unet3d_fold4_best.pth
```

Training histories and validation results are also saved for each fold.

---

## Training Visualizations

The baseline notebook generates training and validation curves for:

- combined loss;
- Dice;
- IoU.

These plots are used to inspect convergence and possible overfitting.

Example:

```text
Training Loss
Validation Loss
Training Dice
Validation Dice
Training IoU
Validation IoU
```

The notebook also provides qualitative segmentation visualizations comparing:

```text
FLAIR
Ground Truth
3D U-Net Prediction
Prediction Overlay
```

---

## Reproducibility

A fixed random seed is used for the experiments:

```python
SEED = 42
```

The same `5fold_manifest.csv` should be reused when comparing different supervised segmentation models.

This ensures that differences between models are not caused by different patient splits.

---

## Planned Model Comparison

The 3D U-Net is used as the baseline for the broader project.

The planned model comparison includes:

1. **3D U-Net baseline**
2. **nnU-Net v2**
3. **Swin UNETR**
4. **MAE-based approach**

The purpose of the baseline is to establish a reference performance before evaluating more advanced architectures and representation-learning approaches.

---

## Environment

The experiments are designed to run in a GPU-enabled environment such as Kaggle.

Main dependencies include:

```text
Python
NumPy
Pandas
NiBabel
PyTorch
scikit-learn
Matplotlib
tqdm
```

---

## Dataset

The MRI dataset is not included in this repository.

The notebooks expect the dataset and metadata to be available in the corresponding Kaggle environment.

Paths used in the notebooks should therefore be adapted to the local execution environment if the project is run elsewhere.

---

## Academic Project

This repository was developed as part of the **AI in Biomedicine** course project.

The repository is intended for research and educational purposes, with emphasis on reproducible patient-level evaluation of volumetric glioma segmentation models.
