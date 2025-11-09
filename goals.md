# Facial Emotion Recognition Project

[![Status](https://img.shields.io/badge/Status-Planned-blue)](https://github.com/yourusername/facial-emotion-local)
[![Dataset](https://img.shields.io/badge/Dataset-FER--2013-green)](https://www.kaggle.com/datasets/msambare/fer2013)
[![Accuracy Target](https://img.shields.io/badge/Accuracy-%3E65%25-orange)](https://arxiv.org/abs/2103.12496)
[![Tech Stack](https://img.shields.io/badge/Tech-PyTorch%2C%20Flask%2C%20OpenCV-brightgreen)](https://pytorch.org/)
[![Estimated Effort](https://img.shields.io/badge/Effort-20--30%20hours-lightgrey)](https://linear.app/)

## Problem Statement

Facial emotion recognition aims to automatically detect and classify human emotions from facial expressions in real-time video streams, such as webcam inputs. This project addresses the challenge of building an end-to-end system for detecting emotions like anger, disgust, fear, happiness, sadness, surprise, and neutral using the FER-2013 dataset, which consists of 48x48 grayscale images reflecting real-world variability in lighting, poses, and expressions.

The focus of this project is on emulating a production environment on a local machine while at the same time adhering to the constraints of such a small system.

The system must handle real-time inference while maintaining robustness to noise, such as poor lighting or occlusions, to simulate production challenges in human-computer interaction (HCI) applications. This involves integrating computer vision for face detection, deep learning for classification, and web technologies for live streaming.

## Methods

The methodology is designed to follow ML engineering best practices, adapted for local execution: modular pipelines for data handling, iterative model development with techniques like transfer learning, and lightweight deployment using an API.

Data preparation uses the FER-2013 dataset (35,887 images across 7 emotion classes), exploration with matplotlib for class distribution analysis, and preprocessing to 48x48 grayscale with torchvision transforms (e.g., normalization, augmentation via flips and crops to mitigate overfitting).

Model development centers on a custom CNN built with PyTorch, featuring convolutional layers, pooling, dropout, and optional transfer learning from ResNet18 (pretrained on ImageNet, adapted for grayscale input) to leverage hierarchical features while keeping parameter counts low for local compute.

Evaluation includes sklearn metrics and visualization of errors/confusion matrices; deployment uses Flask for a web app with OpenCV for webcam capture and Haar cascades for face detection, targeting real-time processing.

Extensions incorporate quantization for inference speedup and basic logging for drift monitoring, reflecting production observability without cloud dependencies.

## Metrics and Goals

### Core Metrics
- **Classification Accuracy**: Overall >65% on FER-2013 test set, surpassing human baseline (~65.5%) and aligning with state-of-the-art single-network benchmarks (e.g., 73% with VGG/ResNet).

- **Per-Class F1-Score**: Weighted average >0.60, with emphasis on underrepresented classes (fear, disgust) to address imbalance; compute via sklearn for bias detection.
- **Real-Time Performance**: Inference latency <200ms per frame, achieving 10-15 FPS on webcam streams using local hardware, measured with OpenCV profiling.
- **Robustness**: >80% accuracy retention under augmentations (e.g., lighting variations); early stopping to prevent overfitting, monitored by val loss divergence.

### (Personal) Proficiency Goals
- **Technical Mastery**: Demonstrate end-to-end ML lifecycle (planning to deployment) with reproducibility (Git, uv for venv, etc.) and modularity (separate train.py, models/emotion_net.py), showcasing first-principles insights like augmentation's role in variance reduction.
- **Optimization and Experimentation**: Conduct hyperparameter tuning (e.g., HyperOpt) and compare baselines (MLP vs. custom CNN vs. transfer learning vs. ViT), logging via MLflow to analyze trade-offs in local setups.
- **Evaluation Depth**: Beyond accuracy, visualize saliency maps (e.g., grad-CAM) and error samples to interpret model weaknesses, benchmarking against random classifiers to justify CNN's spatial feature advantages.
- **Deployment Readiness**: Build a containerized (Docker-optional) Flask app for live emotion detection, testing edge cases (e.g., poor lighting) and extensions like quantization for edge-like efficiency.
- **Learning Outcomes**: Reflect on insights (e.g., why transfer learning outperforms on small datasets) in a final report, targeting 20-30 hours total effort across 7 stages for portfolio depth.

These goals position the project as a senior portfolio piece, emphasizing practical implementation over theoretical knowledge.

## Constraints

- **Hardware Limitations**: Local execution on Apple Silicon MacBook (CPU-only, no GPU acceleration (MPS is available though)), restricting model complexity and batch sizes to avoid excessive training times.
- **Dataset Challenges**: FER-2013's imbalance (e.g., 7,215 happy vs. 436 disgust samples) and noise (low-res 48x48 images, real-world variability) necessitate targeted augmentations without external data sources.
- **Real-Time Requirements**: Webcam integration via OpenCV must handle uncontrolled environments (e.g., varying lighting, no uniform background), with browser security constraints (getUserMedia) limiting seamless streaming.
- **Scope and Tools**: No cloud/distributed systems (e.g., no AWS S3 for data, no Ray for tuning); use open-source local tools (PyTorch, Flask, MLflow, sklearn) and no pre-trained emotion-specific models.
- **Time and Learning Focus**: 20-30 hours cap, prioritizing reflection (e.g., why modular design?) over perfection; no advanced HCI integrations (e.g., AR overlays) to maintain focus on ML proficiency.
