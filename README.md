# GeoFire-Semantic_Fire_Segmentation
# 🔥 Forest Fire Spread Simulation Using AI/ML
## ISRO Bhartiya Antariksh Hackathon 2025

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Data Pipeline & Challenges](#data-pipeline--challenges)
- [Solution Architecture](#solution-architecture)
- [Model Architecture](#model-architecture)
- [Loss Functions](#loss-functions)
- [Training Strategy](#training-strategy)
- [Results & Visualizations](#results--visualizations)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Future Work](#future-work)

---

## 🎯 Problem Statement

Active forest fire detection at **30-meter spatial resolution** using multi-modal satellite data faces a critical challenge: **extreme resolution mismatches** between input modalities. Our objective was to develop a deep learning model that can:

1. Detect active fire boundaries at 30m resolution
2. Fuse coarse climate data (11km ERA5) with high-resolution satellite imagery (30m Sentinel-2)
3. Produce sharp, non-bleeding fire perimeters
4. Generalize across diverse global landscapes

### Resolution Mismatch Challenge
# 🔥 Forest Fire Spread Simulation Using AI/ML
## ISRO Bhartiya Antariksh Hackathon 2025

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Data Pipeline & Challenges](#data-pipeline--challenges)
- [Solution Architecture](#solution-architecture)
- [Model Architecture](#model-architecture)
- [Loss Functions](#loss-functions)
- [Training Strategy](#training-strategy)
- [Results & Visualizations](#results--visualizations)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Future Work](#future-work)

---

## 🎯 Problem Statement

Active forest fire detection at **30-meter spatial resolution** using multi-modal satellite data faces a critical challenge: **extreme resolution mismatches** between input modalities. Our objective was to develop a deep learning model that can:

1. Detect active fire boundaries at 30m resolution
2. Fuse coarse climate data (11km ERA5) with high-resolution satellite imagery (30m Sentinel-2)
3. Produce sharp, non-bleeding fire perimeters
4. Generalize across diverse global landscapes

### Resolution Mismatch Challenge
┌─────────────────────────────────────────────────────────────┐
│ RESOLUTION HIERARCHY │
├──────────────┬──────────────────┬───────────────────────────┤
│ Sentinel-2 │ 30m │ ████████████████████ │
│ NDVI │ │ High-res vegetation │
├──────────────┼──────────────────┼───────────────────────────┤
│ DEM/Slope │ 30m │ ████████████████████ │
│ │ │ High-res terrain │
├──────────────┼──────────────────┼───────────────────────────┤
│ FIRMS │ 1km │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ Labels │ │ Coarse fire detection │
├──────────────┼──────────────────┼───────────────────────────┤
│ ERA5 │ 11km │ ▒▒▒▒▒▒▒▒▒▒ │
│ Climate │ │ Very coarse climate │
└──────────────┴──────────────────┴───────────────────────────┘

**Key Insight:** Traditional approaches that simply upsample coarse data fail because:
- ERA5 temperature (11km) appears as a **flat value** across a 10km patch
- FIRMS labels (1km) produce **blocky, unrealistic** fire boundaries
- Model cannot learn fine-grained boundaries from coarse supervision

---

## 📊 Data Pipeline & Challenges

### Data Sources

| Channel | Source | Resolution | Description |
|---------|--------|------------|-------------|
| Elevation (DEM) | SRTM/ALOS | 30m | Terrain height |
| Slope | Derived from DEM | 30m | Terrain steepness |
| NDVI | Sentinel-2 | 30m | Vegetation health index |
| Temperature | ERA5 | 11km | 2m air temperature |
| Soil Moisture | ERA5 | ~9km | Surface soil moisture |
| Fire Labels | FIRMS | 1km | MODIS/VIIRS active fire |

### Data Extraction Architecture
┌──────────────────────────────────────────────────────────────┐
│ DATA PIPELINE FLOW │
└──────────────────────────────────────────────────────────────┘

Google Earth Engine
│
├──► Sentinel-2 (10-day composites) ──► NDVI @ 30m
│
├──► SRTM DEM ────────────────────────► Elevation @ 30m
│ │
├──► ALOS DSM ────────────────────────► Slope @ 30m
│
├──► ERA5-Land Hourly ──────► Temperature @ 11km
│ Soil Moisture @ 9km
│
└──► FIRMS (MODIS+VIIRS) ───► Active Fire @ 1km

▼
┌─────────────────────┐
│ 10km × 10km Patches │
│ 6-Channel GeoTIFFs │
└─────────────────────┘
│
▼
┌─────────────────────┐
│ PyTorch DataLoader │
│ + Synthesis Pipeline│
└─────────────────────┘
### Challenge 1: Coarse Climate Data

**Problem:** ERA5 temperature at 11km resolution provides only **1 value per 10km patch**. When upsampled, the entire image has identical temperature, providing no spatial gradient for the model to learn from.
ERA5 Temperature (11km) Upsampled to 30m
┌─────────────────┐ ┌────────────────────┐
│ │ │ 27.3 27.3 ... 27.3 │
│ 27.3°C │ ────► │ 27.3 27.3 ... 27.3 │
│ │ │ ... ... ... ... │
└─────────────────┘ │ 27.3 27.3 ... 27.3 │
└────────────────────┘
(Flat - no information!)
