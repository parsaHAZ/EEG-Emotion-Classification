# 🔥 EEG Emotion/Stress Classification

**Bachelor's Degree Project**

This repository contains my **Bachelor's Degree project**, which
implements a full deep-learning pipeline for **EEG-based emotional and
stress state classification**.\
The project processes raw EEG signals, cleans them, segments them into
windows, trains neural models, and evaluates their ability to detect
mental states.

------------------------------------------------------------------------

# 🚀 Project Overview

### 🎓 Academic Context

This project was developed as part of my **Bachelor's Thesis**, focusing
on machine learning for brain--computer interfaces (BCI) and affective
computing.

### 🎯 Objective

**Detect human emotional or stress states from raw EEG brainwave
recordings.**

The project follows a research-grade EEG ML workflow:

1.  Load EEG dataset
2.  Clean and standardize signals
3.  Slice signals into windows/epochs
4.  Normalize numerical features
5.  Encode emotion/stress labels
6.  Split into training & testing sets
7.  Train deep learning models (LSTM / CNN /DNN)
8.  Evaluate performance
9.  Visualize training and results

------------------------------------------------------------------------

# 📡 Data Pipeline

### ✔ 1. EEG Dataset Loading

The notebook loads multi-channel EEG recordings (e.g., AF7, AF8, TP9,
TP10) along with labels representing emotional or stress states.

------------------------------------------------------------------------

### ✔ 2. Signal Preprocessing

Includes: - Filtering & smoothing
- Normalization
- Artifact reduction
- Signal length standardization

Ensures high-quality input for the neural network.

------------------------------------------------------------------------

### ✔ 3. Sliding Window Segmentation

EEG is segmented into fixed-time windows (e.g., 1-second).\
Each window becomes a training sample, boosting dataset size and
capturing temporal structure.

------------------------------------------------------------------------

### ✔ 4. Label Encoding

Example:

    neutral → 0  
    positive → 1  
    negative → 2

------------------------------------------------------------------------

### ✔ 5. Train/Test Split

Uses `train_test_split()` to ensure fair model evaluation.

------------------------------------------------------------------------

# 🧠 Deep Learning Model

### **Main Architecture: LSTM**

LSTMs are ideal for EEG due to temporal memory and sequence modeling.

Model components include: - LSTM layers
- Fully connected layers
- Softmax output
- Dropout
- Adam optimizer + Cross-entropy loss
- EarlyStopping

CNN and DNN options may also be available.

------------------------------------------------------------------------

# 📊 Training Process

Training includes: - Multi-epoch sequence learning
- Automatic stopping when performance stagnates
- Accuracy & loss tracking
- Optional learning-rate scheduling

------------------------------------------------------------------------

# 🎯 Evaluation

The project produces the following metrics: - Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Full Classification Report

------------------------------------------------------------------------

# 📈 Visualizations

Plots include: - Accuracy vs. Epoch
- Loss vs. Epoch
- Optional: EEG signal previews

Helpful for diagnosing overfitting or training instability.

------------------------------------------------------------------------

# 🔧 Requirements

    python >= 3.8  
    numpy  
    pandas  
    matplotlib  
    scikit-learn  
    tensorflow
    pytorch

Install with:

    pip install -r requirements.txt

------------------------------------------------------------------------

# ▶️ How to Run

1.  Install dependencies

2.  Open the notebook:

        jupyter notebook "Bachelor's_Project.ipynb"

3.  Run all cells

4.  View generated metrics and plots

5.  Tune model parameters if desired

------------------------------------------------------------------------

# 🧩 Folder Contents

-   **Bachelor's_Project.ipynb**\
    Complete data pipeline, training code, and evaluation.

-   **papers/JPT_2\_LAKSHMINARAYANA+KODAVALI_6\_1968.pdf**\
[📄 Download the referenced paper](papers/JPT_2_LAKSHMINARAYANA+KODAVALI_6_1968.pdf)\
    Academic reference paper used for theoretical background.

------------------------------------------------------------------------

# 🌟 Final Notes

This work represents a full end-to-end implementation of EEG-based
emotional state classification at a **Bachelor's degree research
level**, closely following established academic EEG processing
standards.
