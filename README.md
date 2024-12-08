# Front-camera-image-segmentation-for-autonomous-driving
Metoak Technology Co., Ltd summer intern

**Overview**
This repository contains the core implementation of a semantic segmentation model for autonomous driving applications. The project focuses on segmenting critical visual elements such as lane markings, vehicles, and signs from front-facing camera footage.

The primary model used for segmentation is a U-Net multiclass image classification model, which is widely adopted for its effectiveness in pixel-level predictions and semantic segmentation tasks.

**Files and Structure**
segmentation.py:
Contains the implementation of the U-Net-based semantic segmentation model. The model is designed to classify and segment various essential objects required for safe and efficient autonomous driving, including:
  - Lane markings
  - Vehicles
  - Road signs

Other Files:
Additional files related to the following have been excluded from this repository due to confidentiality and proprietary restrictions of the company:
  - Company camera hardware interaction scripts
  - Prototype vehicle integration code
  - Proprietary datasets and pre-processing scripts

**Technologies Used**
Deep Learning Framework: PyTorch
Model: Multichannel U-Net multiclass image segmentation
Language: Python
Hardware Requirements: NVIDIA GPU and CUDA support
