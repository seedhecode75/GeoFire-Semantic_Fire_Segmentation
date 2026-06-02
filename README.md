# 🔥 Forest Fire Spread Simulation Using AI/ML
## ISRO Bhartiya Antariksh Hackathon 2025

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Data Pipeline & Synthesis](#-data-pipeline--synthesis)
- [Model Architecture](#-model-architecture)
- [Loss Functions](#-loss-functions)
- [Training Strategy](#-training-strategy)
- [Results & Visualizations](#-results--visualizations)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement
Active forest fire detection at **30-meter spatial resolution** using multi-modal satellite data faces a critical challenge: **extreme resolution mismatches** between input modalities. Our objective was to develop a deep learning model that can:
1. Detect active fire boundaries at 30m resolution.
2. Fuse coarse climate data (11km ERA5) with high-resolution satellite imagery (30m Sentinel-2).
3. Produce sharp, non-bleeding fire perimeters.
4. Generalize across diverse global landscapes.

### The Resolution Mismatch Challenge
| Modality | Native Resolution | Data Type | Constraint |
| :--- | :--- | :--- | :--- |
| **Sentinel-2** | 30m | NDVI | High-res vegetation |
| **SRTM / ALOS** | 30m | DEM & Slope | High-res terrain |
| **FIRMS** | 1km (1000m) | Active Fire Label | **Blocky, coarse targets** |
| **ERA5** | 11km (11000m) | Temp & Moisture | **Featureless regional data** |

**Key Insight:** Traditional approaches that simply upsample coarse data fail. ERA5 temperature (11km) appears as a flat, featureless block across a 10km patch. FIRMS labels (1km) produce 90-degree Minecraft-like fire boundaries. The model cannot learn fine-grained boundaries from raw coarse supervision.

---

## 📊 Data Pipeline & Synthesis

To overcome the resolution mismatch, we engineered a custom PyTorch `DataLoader` that physically synthesizes and refines the data prior to training.

```mermaid
graph TD
    A[Sentinel-2] -->|30m| B(NDVI)
    C[SRTM/ALOS] -->|30m| D(DEM & Slope)
    E[ERA5] -->|11km| F(Temp & Soil Moisture)
    G[FIRMS] -->|1km| H(Active Fire Label)
    B --> I
    D --> I
    F --> I
    H --> I
    I[6-Channel 10x10km GeoTIFF] --> J{PyTorch DataLoader}
    J -->|Physics Synthesis| K(Target Snapping & Thermal Bloom)
    K --> L((Model Input))
