# Wine Quality Prediction

### End-to-End Machine Learning Classification System

The **Wine Quality Prediction System** is an end-to-end machine learning project for predicting wine quality using physicochemical properties.

The system classifies wine into three quality categories:

- **Low**
- **Medium**
- **High**

Rather than focusing only on model training, the project implements a complete and reproducible machine learning workflow covering:

- data ingestion;
- data validation;
- preprocessing;
- feature engineering;
- feature selection;
- class balancing;
- model training;
- model comparison;
- evaluation;
- experiment tracking;
- model persistence;
- REST API deployment.

The project demonstrates how a machine learning experiment can be structured as a **modular, reusable, and deployment-oriented ML system**.

---

# Project Motivation

Machine learning projects often begin inside notebooks, where preprocessing, feature engineering, model training, and evaluation are tightly coupled.

While notebooks are useful for experimentation, this approach can make it difficult to:

- reproduce experiments;
- compare models fairly;
- reuse preprocessing logic;
- track configurations;
- deploy trained models;
- maintain the codebase;
- test individual components.

This project addresses those challenges by organizing the entire machine learning lifecycle into independent modules and a configurable pipeline.

The goal is to move from:

```text
Dataset → Notebook → Model
```

toward:

```text
Data
  ↓
Validation
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Comparison
  ↓
Champion Model
  ↓
REST API
```

---

# Key Features

## Modular ML Pipeline

The project separates major machine learning responsibilities into dedicated modules.

```text
Data Management
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Evaluation
      ↓
Model Comparison
      ↓
Model Persistence
      ↓
API Deployment
```

This makes the pipeline easier to maintain, test, and extend.

---

## Multi-Model Training

The pipeline supports comparison across several machine learning algorithms, including:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Each model can be enabled, disabled, or configured through YAML configuration files.

---

## Comprehensive Evaluation

Models are evaluated using multiple classification metrics rather than relying only on accuracy.

Evaluation includes:

- Accuracy
- Macro F1 Score
- Weighted F1 Score
- Precision
- Recall
- Class-level metrics
- Confusion Matrix
- ROC curves using One-vs-Rest evaluation

This is particularly important for multiclass classification because overall accuracy alone may hide poor performance on minority classes.

---

## Reproducible Experiments

Each pipeline run stores information about the experiment.

Generated artifacts include:

```text
runtime_info.json
config_used.yaml
metrics/*.json
figures/*
```

This allows experiments to be traced back to:

- configuration parameters;
- Python environment;
- package versions;
- execution timestamp;
- preprocessing settings;
- model parameters;
- evaluation results.

---

## Configuration-Driven Pipeline

Pipeline behaviour is controlled through YAML rather than hardcoded parameters.

Users can configure:

- random seed;
- train/test split;
- validation split;
- feature engineering;
- feature-selection methods;
- sampling strategies;
- model selection;
- model hyperparameters.

This makes experiments easier to reproduce and compare.

---

## API Deployment

The trained champion model can be exposed through a **FastAPI REST API**.

The API accepts wine physicochemical properties and returns:

- predicted quality class;
- class probabilities;
- model information;
- prediction timestamp.

FastAPI also automatically provides interactive Swagger documentation.

---

# Dataset

The pipeline works with the Wine Quality datasets containing physicochemical properties of red and white wine.

Typical input variables include:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The original numeric wine-quality score is transformed into three classification categories:

```text
Low
Medium
High
```

This converts the original prediction problem into a **multiclass classification task**.

---

# Project Structure

```text
wine_quality_project/
│
├── src/
│   │
│   ├── data_management/
│   │   ├── downloader.py
│   │   ├── loader.py
│   │   └── processor.py
│   │
│   ├── features/
│   │   ├── engineering.py
│   │   ├── selection.py
│   │   └── transformation.py
│   │
│   ├── models/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── comparator.py
│   │
│   ├── visualization/
│   │   ├── eda_plot.py
│   │   └── model_plots.py
│   │
│   ├── pipeline/
│   │   └── main.py
│   │
│   └── api/
│       └── app.py
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── metrics/
│   ├── figures/
│   └── runs/
│
├── tests/
│
├── notebooks/
│
├── bin/
│   └── run_pipeline.sh
│
├── requirements.txt
├── requirements.lock.txt
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

---

# Machine Learning Workflow

## 1. Data Management

The data-management layer handles ingestion and validation.

It can load datasets such as:

```text
winequality-red.csv
winequality-white.csv
```

Responsibilities include:

- dataset loading;
- download fallback;
- schema validation;
- missing-value handling;
- type validation;
- preprocessing;
- outlier handling.

The objective is to ensure that downstream machine learning components receive clean and predictable input data.

---

# 2. Target Transformation

The original wine-quality variable contains numerical scores.

For this project, the target is converted into three classes:

```text
Low
Medium
High
```

This creates a multiclass classification problem that can be evaluated using class-sensitive metrics such as macro F1.

---

# 3. Feature Engineering

The feature-engineering pipeline supports transformations such as:

- interaction features;
- polynomial features;
- correlation-based feature selection;
- statistical feature selection;
- feature scaling;
- normalization.

The pipeline is configurable, so feature-engineering strategies can be enabled or disabled between experiments.

---

# 4. Class Imbalance Handling

Wine-quality classes may not appear equally frequently.

The pipeline therefore supports optional class-balancing techniques such as:

**SMOTE — Synthetic Minority Oversampling Technique**

This allows experiments to compare models trained with and without oversampling.

Importantly, sampling is applied as part of the training workflow rather than directly to the complete dataset to reduce the risk of data leakage.

---

# 5. Model Training

The training layer provides a common interface for fitting multiple machine learning models.

Supported algorithms include:

```text
Logistic Regression
Support Vector Machine
Random Forest
Gradient Boosting
XGBoost
LightGBM
```

This makes it possible to compare different modelling approaches under consistent preprocessing and evaluation conditions.

---

# 6. Model Evaluation

Each trained model is evaluated using a range of metrics.

## Accuracy

Measures the overall proportion of correct predictions.

## Precision

Measures how many predictions assigned to a class were correct.

## Recall

Measures how many actual members of a class were correctly identified.

## Macro F1

Calculates the F1 score independently for each class and gives each class equal importance.

This is useful for evaluating performance when class distributions are imbalanced.

## Weighted F1

Calculates F1 while weighting each class according to its frequency.

## Confusion Matrix

Provides a class-by-class view of prediction errors.

## ROC Analysis

Multiclass ROC evaluation uses a **One-vs-Rest (OVR)** strategy.

---

# 7. Model Comparison

Model comparison is handled through:

```text
src/models/comparator.py
```

Models are ranked according to the configured selection metric.

For example:

```text
f1_weighted
```

The best-performing model becomes the **champion model** and can be persisted for deployment.

Example:

```text
models/champion.joblib
```

---

# 8. Experiment Tracking

Every experiment automatically stores metadata and outputs.

Example run directory:

```text
reports/runs/2025-10-07/
│
├── config_used.yaml
├── runtime_info.json
│
├── metrics/
│   ├── logistic_regression.json
│   ├── random_forest.json
│   └── xgboost.json
│
└── figures/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── model_comparison.png
```

This improves reproducibility because every result can be linked back to the configuration used to generate it.

---

# Running the Project

## 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd wine_quality_project
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv winequality
winequality\Scripts\activate
```

### Linux / macOS

```bash
python -m venv winequality
source winequality/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

To save an exact snapshot of the environment:

```bash
pip freeze > requirements.lock.txt
```

---

# Running the ML Pipeline

The complete pipeline can be executed using:

```bash
bash bin/run_pipeline.sh
```

or directly with Python:

```bash
python -m src.pipeline.main \
  --config configs/default.yaml \
  --outdir reports/runs/local
```

After execution, outputs are stored under:

```text
reports/runs/
```

including:

- model metrics;
- visualizations;
- runtime metadata;
- configuration snapshots;
- trained-model information.

---

# Configuration

Example:

```yaml
random_state: 42
test_size: 0.2
val_size: 0.2
target_col: quality_category

preprocess:
  create_interactions: true
  feature_selection_method: correlation

models:
  logistic_regression:
    enabled: true

  random_forest:
    enabled: true

  svm:
    enabled: true
```

Configurations allow experiments to be modified without changing pipeline code.

---

# REST API Deployment

Once the champion model has been generated, start the prediction API with:

```bash
uvicorn src.api.app:app --reload
```

The server will normally run at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# Prediction Endpoint

The main prediction endpoint is:

```text
POST /predict
```

Example request:

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.70,
  "citric_acid": 0.00,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

Example response:

```json
{
  "prediction": "Medium",
  "probabilities": {
    "Low": 0.10,
    "Medium": 0.80,
    "High": 0.10
  },
  "model": "champion.joblib",
  "timestamp": "2025-10-07T16:22Z"
}
```

---

# Evaluation Results

The model-comparison process selects the best-performing model according to the configured evaluation metric.

A result summary can be presented as:

| Model | Accuracy | Macro F1 | Weighted F1 | Status |
|---|---:|---:|---:|---|
| Random Forest | 0.89 | 0.87 | 0.88 | Champion |

> **Note:** The values above are example values. Replace them with the actual output from `reports/metrics/test_metrics_champion.json` before presenting them as project results.

This is important because the repository should report only experimentally verified performance.

---

# Technology Stack

## Programming

- Python 3.12+

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-Learn

## Model Evaluation

- Scikit-learn metrics
- Multiclass ROC analysis
- Confusion matrices

## Visualization

- Matplotlib
- Seaborn

## API

- FastAPI
- Uvicorn

## Model Persistence

- Joblib

## Configuration & Experiment Tracking

- YAML
- JSON

## Software Engineering

- Git
- GitHub
- Unit testing
- Modular Python packages
- Docker

---

# Engineering Principles

This project was designed around several machine learning engineering principles.

### Separation of Concerns

Data preparation, feature engineering, training, evaluation, visualization, and deployment are implemented separately.

### Reproducibility

Pipeline configurations and environment information are stored for each run.

### Configuration over Hardcoding

Experiment settings are controlled through YAML files.

### Consistent Evaluation

Models are compared using the same preprocessing and evaluation pipeline.

### Model Persistence

The selected champion model is saved so that the exact trained model can be reused during inference.

### Deployment Readiness

The model is exposed through an API rather than remaining only inside a notebook.

---

# Why This Project Matters

The predictive task itself is relatively straightforward.

The primary objective of this project is therefore not simply to achieve the highest possible wine-classification accuracy.

The project demonstrates how to take a machine learning problem through a broader engineering lifecycle:

```text
Raw Data
    ↓
Reusable Processing
    ↓
Feature Engineering
    ↓
Multiple ML Models
    ↓
Reproducible Evaluation
    ↓
Champion Selection
    ↓
Model Persistence
    ↓
API Deployment
```

This makes the project an example of **machine learning system design**, rather than only model experimentation.

---

# Future Improvements

Potential extensions include:

- SHAP-based model explanations;
- global and local feature-importance visualizations;
- `/metadata` API endpoint;
- `/health` API endpoint;
- controlled model retraining;
- MLflow experiment tracking;
- automated hyperparameter optimization;
- model and data-drift monitoring;
- CI/CD-based testing and deployment;
- Dockerized API deployment;
- cloud deployment using AWS or Azure;
- automated model-performance monitoring.

---

# Author

**Regina Adobea Essien**

Data Scientist & AI Researcher

Email: [reginaessien83@gmail.com](mailto:reginaessien83@gmail.com)

LinkedIn: [Adobea Essien](https://www.linkedin.com/in/adobea-essien/)

GitHub: [adobea-dev](https://github.com/adobea-dev)
