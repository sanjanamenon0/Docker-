# Docker Lab 1 — Containerized ML Model Training

## Overview

This lab containerizes a Random Forest classifier trained on the Iris dataset using Docker. The original lab trains the model and saves it as a `.pkl` file. This version replicates that baseline and adds three additional functionalities reflecting real MLOps practices.

## Project Structure

LAB1/
├── src/
│   ├── main.py
│   └── requirements.txt
├── Dockerfile
└── README.md


## Build and Run
```bash
docker build -t lab1:v1 .
docker run lab1:v1
```

Save the image:
```bash
docker save lab1:v1 > my_image.tar
```

Copy the training summary out of the container:
```bash
$id = docker ps -aq | Select-Object -First 1
docker cp ${id}:/app/training_summary.json .
```

---

## Added Features

### 1. Model Evaluation Metrics
Evaluates the model on the held-out test set and prints accuracy and a classification report (precision, recall, F1 per class). The original lab gives zero feedback on model performance — this makes every training run measurable and auditable.

### 2. Feature Importance Logging
Extracts and ranks each feature's contribution to the model's predictions. Petal dimensions account for ~86% of importance, confirming the model is learning meaningful patterns. Useful for interpretability and detecting data drift across runs.

### 3. Training Summary Export (JSON)
Saves a `training_summary.json` file inside the container with model type, accuracy, timestamp, and feature importances. This is the foundation of experiment tracking — every run produces a persistent, machine-readable record that can be versioned and compared across training jobs.

---

## Dependencies
`scikit-learn`, `joblib` — `numpy`, `json`, `datetime` are either pulled in automatically or part of Python's standard library.
