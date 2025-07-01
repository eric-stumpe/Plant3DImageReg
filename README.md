# Plant3DImageReg

This repository contains the source code and dataset for our publication (download dataset from: https://zenodo.org/records/15782840):

📄 **[3D multimodal image registration for plant phenotyping](https://doi.org/10.1016/j.compag.2025.110538)**  
Published in *Computers and Electronics in Agriculture*, 2025.

## Overview

This project presents an image registration method for multimodal cameras (e.g., thermal, hyperspectral, infrared, RGB) using depth data and ray casting. Unlike traditional image-based approaches, our method compensates for parallax, occlusions, and errors by incorporating 3D scene information.

### Key Features

- Supports arbitrary numbers of fixed-position cameras.
- Registers images across different modalities (thermal, hyperspectral, etc.).
- Uses depth information for accurate 3D alignment and point cloud generation.
- Creates filter masks for occlusion and uncertainty areas
- Includes code for both **camera calibration** and **image registration**.

---

## 📌 Use Case Checklist

Before using this pipeline, verify that your imaging setup meets the following requirements:

- **Fixed Multi-Camera Configuration:**  
  Your system must consist of two or more cameras whose positions relative to each other remain constant. The number and type of cameras are arbitrary, but they must be mounted in a rigid configuration (e.g., on a frame or gantry). Movements of the entire rig are allowed, but the internal geometry must not change during recording. Otherwise the calibration has to be repeated.

- **Multimodal Compatibility:**  
  The method supports diverse imaging modalities. In our experiments, we used the following, but other modalities are possible:
  - Thermal camera
  - Hyperspectral camera
  - Infrared (IR) camera
  - RGB camera

- **Calibration Feasibility for All Modalities:**  
  You must be able to capture calibration images (e.g., checkerboard patterns) with every camera. Some modalities require specific techniques:
  - In our publication the thermal images had to be be prepared by heating up the black tiles with a halogen lamp.
  - Other modalities such as Fluorescence or NIR imaging we did not test may require custom calibration targets or lighting setups.

- **Depth Sensor Availability:**  
  Registration relies on 3D scene structure. You must have a depth-sensing device (e.g., LiDAR, depth camera) that provides depth information of the same scene.

- **Depth and IR Alignment Required:**  
  In our implementation, we used the **Microsoft Azure Kinect**, which outputs:
  - A **depth image**
  - An **infrared image** that is spatially aligned with the depth image  
  This alignment is critical, as we perform extrinsic calibration between this IR image and all other camera views to relate them to the 3D geometry.

- **Alternative Depth Solutions:**  
  If your depth sensor **does not** produce an aligned image (e.g., LiDAR scanning), you will have to extend the pipeline:
  - Use a 3D calibration object (e.g., ArUco 3D cube or point-based calibration target).
  - Compute extrinsics between 2D camera images and 3D depth geometry via a suitable calibration method.

---

## 🌱 Dataset Description

The dataset used in this repository (including both plant and calibration images) was recorded on **January 26th, 2024** at the **UFT Tulln (BOKU), Austria** 
and is available at: https://zenodo.org/records/15782840.

It includes multimodal images captured from:
- RGB/Depth/Infrared camera (Azure Kinect)
- Thermal camera (Infratec VarioCAM HDx Head 620 S)
- Hyperspectral camera (Specim IQ)

These images are used for camera calibration and multimodal image registration as described in our publication.  
📄 [Read more about the data acquisition process](https://doi.org/10.1016/j.compag.2025.110538)

---

## 🗂️ Repository Structure

### Files

| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies (tested with Python 3.12). |
| `01calibrate.ipynb` | Jupyter notebook for intrinsic/extrinsic calibration using checkerboards. |
| `02register.ipynb` | Jupyter notebook for multimodal image registration and point cloud generation. |
| `utils.py` | Helper functions used by both notebooks. |
| `wavebands.json` | Hyperspectral channel metadata (204 bands from Specim IQ). |

### Folders

| Folder | Description |
|--------|-------------|
| `calibration_dataset/` | Calibration images for each modality (Download and unzip from: https://zenodo.org/records/15782840)|
| └─ `infra/` | PNG IR images (Azure Kinect) |
| └─ `rgb/` | PNG RGB images (Azure Kinect) |
| └─ `thermal/` | PNG thermal images (VarioCAM HDx Head 620 S) |
| └─ `hs/` | PNG RGB representations from hyperspectral camera (Specim IQ) |
| `image_dataset/` | Plant images ((Download and unzip from: https://zenodo.org/records/15782840) |
| └─ `infra/` | `.npy` IR images |
| └─ `rgb/` | `.png` RGB and `.npy` depth (Azure Kinect) |
| └─ `thermal/` | `.npy` thermal images |
| └─ `hs_full/` | Full 512×512×204 hyperspectral `.npy` files |
| └─ `hs_rgb/` | RGB representations of hyperspectral images |
| └─ `hs_pca/` | PCA false color visualizations |
| `3D_pointclouds/` | Output folder for registered point clouds |
| `3D_meshes/` | Output folder for object and uncertainty meshes |

---

## ⚙️ Installation Guide

We used [Miniforge](https://conda-forge.org/download/) to manage dependencies via Conda, but you are free to use any other program.

### 1. Install Miniforge

Update the base environment:

```bash
conda update --all
```

### 2. Create a virtual environment

```bash
conda create -n 3Dregenv python=3.12
conda activate 3Dregenv
```

### 3. Clone this repository and install dependencies

```bash
cd path/to/your/cloned/repo
pip install -r requirements.txt
```

Alternatively, you may install packages individually:

```bash
pip install numpy matplotlib imageio opencv-python jupyter open3d
```

---

## 🚀 Usage

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

2. Run:
   - `01calibrate.ipynb` to compute and save camera calibration parameters.
   - `02register.ipynb` to register multimodal images and generate point clouds/meshes.

---

## 📦 Data Download

⚠️ The following folders are placeholders in this repo and must be downloaded from **Zenodo**:

- `calibration_dataset/`
- `image_dataset/`

Please replace the folders with the actual data from the Zenodo repository linked in the paper.

---

## 📫 Citation

If you use this repository in your research, please cite:

```bibtex
@article{stumpe20253d,
  title={3D multimodal image registration for plant phenotyping},
  author={Stumpe, Eric and Bodner, Gernot and Flagiello, Francesco and Zeppelzauer, Matthias},
  journal={Computers and Electronics in Agriculture},
  volume={237},
  pages={110538},
  year={2025},
  publisher={Elsevier}
}
```

## 🧾 Acknowledgments

This research was funded in whole by the **Research Promotion Agency of Lower Austria (GFF)** through the project:

**“DiPhSpec — Digitizing physiology for advanced spectral plant diagnostics”**  
(Project No. FTI18-005)

Additionally, this work was supported by the **PhD scholarship ‘‘Visual Monitoring of Crop Production Systems:
A Multimodal Approach’’ (project no. FTI21-D-021).**

