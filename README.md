# TrainForge

A lightweight machine learning server that supports **asynchronous model training** and **fast inference**.
The system isolates training workloads in separate processes while keeping inference fast using memory-mapped models.

This project demonstrates core ideas used in real ML serving systems:

* Supervisor process managing jobs
* Trainer process pool
* Job queue for scheduling training tasks
* Resource limits for training
* Persistent models stored on disk
* Fast inference using `mmap`

---

# Architecture

The system consists of three main components.

### Supervisor (FastAPI Server)

Responsibilities:

* Accept training requests
* Push jobs into the training queue
* Store metadata in SQLite
* Handle inference requests
* Cache loaded models

### Trainer Pool

* Fixed number of worker processes (default: **20**)
* Workers pull jobs from a queue
* Each worker trains models in isolation
* Models are saved to disk after training

### Inference Workers

* Thread pool for concurrent inference
* Models loaded with `mmap`
* Cached in memory for faster predictions

---

# System Flow

### Training

Client → `/train` request
→ Job added to queue
→ Trainer worker picks job
→ Model trained using SVM
→ Model saved to `model_store/`
→ Status updated in SQLite

### Inference

Client → `/infer` request
→ Worker thread handles request
→ Model loaded using `mmap`
→ Prediction returned

---

# Project Structure

```
project/
│
├── server.py
├── trainer_worker.py
├── metadata.db
├── model_store/
│
├── data/
│   └── train.csv
│
└── README.md
```

---

# Installation

Install dependencies:

```
pip install fastapi uvicorn scikit-learn numpy joblib
```

---

# Running the Server

Start the server:

```
python server.py
```

or

```
uvicorn server:app --port 8000
```

The server will automatically start the **trainer process pool**.

---

# Dataset Format

Training dataset must be a CSV file where:

* Each row is a sample
* Last column is the label

Example:

```
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
6.4,3.2,4.5,1.5,1
5.9,3.0,4.2,1.5,1
```

---

# API Endpoints

## Train Model

```
POST /train
```

Request:

```
{
  "client_id": "client-1",
  "dataset_path": "data/train.csv",
  "params": {
    "kernel": "linear",
    "C": 1.0
  }
}
```

Response:

```
{
  "token": "model-id",
  "status": "queued"
}
```

---

## Check Training Status

```
GET /status/{token}
```

Query parameter:

```
client_id=client-1
```

Response:

```
{
  "state": "queued | training | completed | failed"
}
```

---

## Run Inference

```
POST /infer
```

Request:

```
{
  "client_id": "client-1",
  "token": "model-id",
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:

```
{
  "prediction": 0
}
```

---

# Key Design Features

### Process Isolation

Training runs in separate processes to prevent:

* memory leaks
* crashes affecting the server

### Training Queue

Limits parallel training jobs and prevents resource overload.

### Resource Limits

Training workers apply:

* memory limits
* CPU time limits

### Memory-Mapped Models

Models are loaded using `mmap`, allowing:

* faster loading
* reduced memory duplication
* shared OS page cache

---

# Possible Improvements

* Model eviction (LRU cache)
* Trainer health monitoring
* Job retry logic
* Multiple model types (XGBoost, Neural Networks)