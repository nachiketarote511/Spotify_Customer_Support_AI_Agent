# Development Setup Guide

## Prerequisites

- Python 3.11+
- pip
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone <repo-url>
cd hiver-ai-support-agent
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Train Models (First Time)

```bash
python scripts/train.py
```

This runs:
- Intent discovery & clustering
- Baseline classifier training
- FAISS embedding index building
- Golden evaluation set creation

### 5. Start the API Server

```bash
python app/api/main.py
```

Visit `http://localhost:8000` for the chatbot UI.

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

### 7. Run Evaluation

```bash
python scripts/run_quick_eval.py
```

## Project Structure

```
hiver-ai-support-agent/
├── backend/              # API + AI core
│   ├── app/             # Application code
│   ├── tests/           # Test suites
│   ├── scripts/         # Training & eval scripts
│   └── requirements.txt
├── frontend/            # Chatbot UI
│   └── public/
├── data-pipeline/       # dbt + data contracts
│   ├── dbt/
│   └── schemas/
├── configs/             # YAML configurations
├── data/                # Processed datasets
├── dataset/             # Raw TWCS dataset
├── models/              # Trained model artifacts
├── reports/             # Evaluation reports
├── docs/                # Documentation
├── docker-compose.yml
└── .env.example
```
