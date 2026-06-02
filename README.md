# 🔥 Forest Fire Prediction- FireUNET
## ISRO Bhartiya Antariksh Hackathon

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

---

## 🎯 Problem Statement
Active forest fire detection at **30-meter spatial resolution** using multi-modal satellite data faces a critical challenge: **extreme resolution mismatches** between input modalities. objective was to develop a deep learning model that can:
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

<img width="5370" height="3577" alt="raw_all" src="https://github.com/user-attachments/assets/63de355e-9228-4b8f-b98f-4dac2e32a15f" />


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
    J -->|Physics Synthesis| K(Target Snapping)
    K --> L((Model Input))
```
<img width="5294" height="3443" alt="ndvi_refinement_pipeline (4)" src="https://github.com/user-attachments/assets/f03c81f3-20b9-4ca7-9081-a429c892d2d6" />

---

# 🧠 Model Architecture

### Multi-Scale Fusion U-Net with Attention

Our architecture extends the standard U-Net with two key innovations:

#### 1. Dual-Stream Encoder
- **High-Resolution Stream:** Processes DEM, Slope, NDVI at native 30m resolution
- **Low-Resolution Stream:** Processes ERA5 climate data at ~1km scale using adaptive pooling
- **Fusion Points:** Low-res features upsampled and concatenated at 4 decoder levels

#### 2. Attention-Gated Skip Connections
Traditional U-Nets concatenate encoder features directly. We use attention gates that learn to focus on relevant spatial regions:

```mermaid
graph TD
    subgraph High-Res[High-Resolution Stream 30m]
        HR_Input["DEM + Slope + NDVI<br/>3×256×256"] --> HR_Inc["Conv 64<br/>256×256"]
        HR_Inc --> HR_D1["MaxPool + Conv 128<br/>128×128"]
        HR_D1 --> HR_D2["MaxPool + Conv 256<br/>64×64"]
        HR_D2 --> HR_D3["MaxPool + Conv 512<br/>32×32"]
        HR_D3 --> HR_D4["MaxPool + Conv 1024<br/>16×16"]
    end

    subgraph Low-Res[Low-Resolution Stream 1km]
        LR_Input["Temperature + Soil Moisture<br/>2×256×256"] --> LR_Pool["AdaptiveAvgPool<br/>32×32 ≈1km"]
        LR_Pool --> LR_E1["Conv 32<br/>32×32"]
        LR_E1 --> LR_E2["MaxPool + Conv 64<br/>16×16"]
        LR_E2 --> LR_E3["Conv 64<br/>8×8"]
    end

    LR_E3 -->|Upsample & Fuse| Fuse4
    LR_E3 -->|Upsample & Fuse| Fuse3
    LR_E3 -->|Upsample & Fuse| Fuse2
    LR_E3 -->|Upsample & Fuse| Fuse1

    subgraph Decoder[Decoder with Attention]
        Fuse4["Fusion 1024+64"] --> Up1["UpConv 512 + Attention Gate"]
        Up1 --> Fuse3["Fusion 512+64"]
        Fuse3 --> Up2["UpConv 256 + Attention Gate"]
        Up2 --> Fuse2["Fusion 256+64"]
        Fuse2 --> Up3["UpConv 128 + Attention Gate"]
        Up3 --> Fuse1["Fusion 128+64"]
        Fuse1 --> Up4["UpConv 64 + Attention Gate"]
    end

    HR_D4 --> Fuse4
    HR_D3 -->|Skip| Up1
    HR_D2 -->|Skip| Up2
    HR_D1 -->|Skip| Up3
    HR_Inc -->|Skip| Up4

    Up4 --> Out["Conv 1×1 → Sigmoid<br/>Fire Mask"]
```

### Model Specifications

| Component | Details |
|-----------|---------|
| High-Res Encoder | 5 levels: 64→128→256→512→1024 |
| Low-Res Encoder | 3 levels: 32→64→64 |
| Decoder | 4 levels with transposed convolutions |
| Attention Gates | 4 gates (one per decoder level) |
| Total Parameters | ~7.8M |
| Input Size | 256×256 pixels (7.68km²) |

# 📉Loss Functions

```mermaid
flowchart LR
    A[Model Output<br/>Logits + Targets] --> B[Sigmoid<br/>Probability Map]
    B --> C[Focal Loss<br/>α=0.75, γ=2.0]
    B --> D[Dice Loss<br/>smooth=1.0]
    B --> E[Edge Loss<br/>Laplacian ∇²]
    B --> F[Positive Penalty<br/>min_prob=0.3]
    
    C -->|0.70| G((Σ))
    D -->|0.30| G
    E -->|0.05| G
    F -->|0.20| G
    
    G --> H[Total Loss]
    
    style C fill:#FF6B6B,color:#fff
    style D fill:#4ECDC4,color:#fff
    style E fill:#45B7D1,color:#fff
    style F fill:#96CEB4,color:#fff
    style G fill:#FFD93D,color:#000
    style H fill:#333,color:#fff
```

# 🎯 Dynamic Threshold Optimization
```mermaid
flowchart TD
    A[Validation Predictions<br/>Probability Map] --> B[Compute Metrics<br/>at Multiple Thresholds]
    B --> C{F1 vs IoU<br/>Optimal Thresholds}
    C --> D[τ_F1 = argmax F1]
    C --> E[τ_IoU = argmax IoU]
    D --> F[F1‑Optimal Model<br/>High recall, more false positives]
    E --> G[IoU‑Optimal Model<br/>High precision, sharper boundaries]
    F --> H[Use Case: Early Warning<br/>Catch all possible fires]
    G --> I[Use Case: Damage Assessment<br/>Accurate area estimation]
    H & I --> J[Deploy appropriate<br/>threshold for the task]
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Epochs | 100 (with early stopping) |
| Optimizer | AdamW |
| Initial LR | 1e-4 |
| LR Schedule | Warmup (10 epochs) + Cosine Annealing |
| Weight Decay | 1e-5 |
| Gradient Clipping | 1.0 |
| Train/Val Split | 80/20 |

# Fire Detection Visualizations

<!-- Add prediction visualizations here -->
<img width="1989" height="985" alt="final_image" src="https://github.com/user-attachments/assets/97870044-ca49-4d25-a2c6-21dad816d85c" />
<img width="1989" height="985" alt="final_2" src="https://github.com/user-attachments/assets/4f53da17-1939-49f1-99e0-d8e464c33e26" />
Figure: Model predictions showing input channels, ground truth, and predicted fire masks
