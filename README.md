<div align="center">

<h1>SDSCNet: Structure-aware Detail-Semantic Collaboration Network for Salient Object Detection in Optical Remote Sensing Images</h1>
</div>

## 📰 News
This project provides the code and results for SDSCNet.

## ⭐ Abstract
Salient object detection in optical remote sensing images (ORSI-SOD) aims to segment the most visually prominent objects in optical remote sensing images (ORSIs). However, existing methods still face notable challenges in the collaborative modeling of heterogeneous features, perception of structurally complex objects, and preservation of boundary integrity. To address these issues, we propose a novel network, Structure-aware Detail-Semantic Collaboration Network (SDSCNet), which integrates three key modules: the Hierarchical Adaptive Perception Module (HAPM), the Structure-Aware Attention Module (SAAM), and the Dynamic Context-aware Edge Refinement Module (DCERM). Specifically, HAPM mitigates the resolution and representation gap between shallow and deep features by introducing a morphology-guided dynamic perception mechanism along with a progressive multi-scale receptive field regulation strategy to achieve targeted enhancement. SAAM, as a dedicated attention module tailored for ORSIs, constructs direction-aware long- and short-range spatial dependencies and incorporates multi-granularity channel representations to adapt to the scale diversity and structural complexity of salient objects. Meanwhile, DCERM extracts multi-scale boundary features adaptively and employs a dual-path fusion strategy to effectively integrate edge cues with high-level semantics, thereby enhancing boundary completeness and contour delineation accuracy in complex scenarios. Extensive experiments on three ORSI-SOD benchmark datasets demonstrate that SDSCNet consistently outperforms 21 state-of-the-art methods. The code is publicly available at https://github.com/Daylight-tyf/SDSCNet.

## 🌏 Network Architecture
   <div align=center>
   <img src="https://github.com/Daylight-tyf/SDSCNet/blob/main/SDSCNet/images/SDSCNet.png">
   </div>
The overall architecture of SDSCNet. It integrates three core modules: the Hierarchical Adaptive Perception Module (HAPM), the Structure-Aware Attention Module (SAAM), and the Dynamic Context-aware Edge Refinement Module (DCERM).

<div align=center>
   <img src="https://github.com/Daylight-tyf/SDSCNet/blob/main/SDSCNet/images/HAPM.png">
   </div>
Illustrations of the proposed HAPM.

<div align=center>
   <img src="https://github.com/Daylight-tyf/SDSCNet/blob/main/SDSCNet/images/SAAM.png">
   </div>
Illustrations of the proposed SAAM.

<div align=center>
   <img src="https://github.com/Daylight-tyf/SDSCNet/blob/main/SDSCNet/images/DCERM.png">
   </div>
Illustrations of the proposed DCERM.
   
## 🖥️ Requirements
   python 3.8 + pytorch 1.9.0
   
## 🚀 Training
   Download [pvt_v2_b2.pth] and put it in './model/'. 
   
   Modify paths of datasets, then run train.py.

Note: Our main model is under './model/SDSCNet.py'

## 🛸 Testing
   1. Modify paths of pre-trained models and datasets.

   2. Run test.py.

## 🖼️ Quantitative comparison
   <div align=center>
   <img src="https://github.com/Daylight-tyf/SDSCNet/blob/main/SDSCNet/images/Table.png">
   </div>
   
## 🌃 Visualization
   <div align=center>
   <img src="https://github.com/Daylight-tyf/SDSCNet/blob/main/SDSCNet/images/Visualization.png">
   </div>
