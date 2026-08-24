# Multi-Class Video Classification using Deep Learning 🎥🧠.

An industry-sponsored Deep Learning project by NVIDIA focused on large-scale multi-class video classification using spatial and temporal learning techniques.

The system was developed end-to-end as part of an academic research project and is capable of classifying videos into four categories:
- Animation
- Gaming
- Natural Content
- Flat Content

## 🚀 Features

- Deep Learning-based video classification pipeline
- Frame-level spatial feature extraction
- Temporal sequence understanding
- Attention-based video representation
- Real-time inference support
- Robust preprocessing and augmentation pipeline
- Out-of-Distribution (OOD) trust scoring
- GPU-accelerated training and inference

## 🧠 Model Architecture

The pipeline combines multiple deep learning components for accurate video understanding:

- **ResNet18 / EfficientNet** for spatial feature extraction
- **Bi-LSTM** for temporal modeling
- **Multi-Head Self Attention** for informative frame selection
- **Model Ensembling** for improved robustness
- **Test-Time Augmentation (TTA)** for better generalization

## ⚙️ Processing Pipeline

The system includes:
- Adaptive frame extraction
- Black-frame filtering
- Blur detection using Laplacian variance
- Image sharpening
- GPU preprocessing
- ImageNet normalization
- Video-level prediction aggregation

## 📊 Results

- Achieved approximately **91–93% classification accuracy**
- Evaluated on a diverse dataset of **3,500+ to 4,000 videos**
- Dataset curated from **YouTube-8M**
- Training performed on **NVIDIA GPU servers (A100 GPUs)**

## 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- CUDA
- Flask
- NVIDIA GPU Infrastructure

## 🌐 Additional Features

- Flask-based inference server
- Video upload + URL-based classification
- Attention visualization
- OOD confidence scoring
- Real-time prediction pipeline

## 👨‍💻 Development

The complete system including:
- Model architecture
- Training pipeline
- Data preprocessing
- Experimentation
- Optimization
- Inference APIs
- Evaluation framework

was developed from scratch as part of an NVIDIA-sponsored academic project.

## 🏫 Institution

Developed at:
- Vishwakarma Institute of Information Technology (VIIT Pune)

## 🌟 Project Goal

To build a scalable and robust deep learning system capable of understanding complex video content using both spatial and temporal information while maintaining high accuracy and real-world deployment capability.
