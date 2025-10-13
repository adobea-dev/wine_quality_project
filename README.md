## 🧠 Wine Quality Prediction – End-to-End ML System

A fully modular machine learning pipeline for multiclass wine quality classification (Low, Medium, High).
This project demonstrates a production-ready ML workflow — from data ingestion and feature engineering to model evaluation, persistence, and REST API deployment.

🚀 Highlights

Automated ML pipeline (data cleaning → feature engineering → model training → evaluation)

Reproducible experiments via YAML configs & runtime logging

Model comparison across Logistic Regression, Random Forest, SVM, XGBoost, and more

Explainable metrics – macro/micro/weighted F1, precision/recall, confusion matrix, ROC (OVR)

FastAPI deployment exposing a /predict endpoint

Config-driven runs: adjustable preprocessing, sampling, and model parameters

📁 Project Structure

wine_quality_project/
├── 📁 src/                          # Source code
│   ├── 📁 data_management/          # Data handling modules
│   │   ├── downloader.py            # Data download with fallback
│   │   ├── loader.py                # Data loading and validation
│   │   └── processor.py             # Data cleaning and preprocessing
│   ├── 📁 features/                 # Feature engineering
│   │   ├── engineering.py           # Feature creation
│   │   ├── selection.py             # Feature selection methods
│   │   └── transformation.py        # Data scaling and normalization
│   ├── 📁 models/                   # ML model components
│   │   ├── trainer.py               # Model training
│   │   ├── evaluator.py             # Model evaluation
│   │   └── comparator.py            # Model comparison
│   ├── 📁 visualization/            # Plotting and visualization
│   │   ├── eda_plot.py              # Exploratory data analysis plots
│   │   └── model_plots.py           # Model performance plots
│   └── 📁 pipeline/                 # Main pipeline orchestration
│       └── main.py                  # Complete ML pipeline
├── 📁 data/                         # Data storage
│   ├── raw/                         # Raw datasets
│   └── processed/                   # Processed datasets
├── 📁 models/                       # Trained model files
├── 📁 reports/                      # Generated reports and plots
├── 📁 tests/                        # Unit tests
├── 📁 notebooks/                    # Jupyter notebooks (optional)
├── app.py                           # Flask web application
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
├── docker-compose.yml               # Docker configuration
└── README.md                        # This file


🧩 Workflow Overview
1️⃣ Data Management

Downloads or loads dataset (winequality-red.csv, winequality-white.csv)

Cleans missing values, fixes types, removes outliers

Converts wine quality scores to multiclass categories (Low / Medium / High)

2️⃣ Feature Engineering

Polynomial and interaction feature generation

Feature selection via correlation or statistical tests

Scaling via StandardScalerWrapper

Optional SMOTE for class balance

3️⃣ Model Training & Evaluation

Trains multiple models (Logistic Regression, SVM, RandomForest, GradientBoosting, XGBoost, LightGBM)

Evaluates using:

Accuracy

Macro/Weighted F1

Precision/Recall per class

Confusion matrix

ROC (One-vs-Rest)

4️⃣ Model Comparison

src/models/model_comparator.py ranks models by f1_weighted

Saves metrics & charts (reports/metrics, reports/figures)

5️⃣ Experiment Tracking

Auto-saves:

runtime_info.json → Python, library versions, OS, timestamp

config_used.yaml → parameters used for that run

metrics/*.json → all performance outputs

6️⃣ Deployment

FastAPI app serves predictions:

uvicorn src.api.app:app --reload


Swagger docs: http://127.0.0.1:8000/docs

⚙️ Reproducible Run

Run the full pipeline and log everything automatically:

bash bin/run_pipeline.sh


Or manually:

python -m src.pipeline.main --config configs/default.yaml --outdir reports/runs/local


Results, metrics, and plots will appear under:

reports/runs/<timestamp>/

🌐 API Usage

After running:

uvicorn src.api.app:app --reload


Visit:

http://127.0.0.1:8000/docs


Example Request:

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


Response:

{
  "prediction": "Medium",
  "probabilities": {"Low": 0.10, "Medium": 0.80, "High": 0.10},
  "model": "champion.joblib",
  "timestamp": "2025-10-07T16:22Z"
}

🧪 Evaluation Summary
Metric	Best Model	Accuracy	Macro F1	Weighted F1
RandomForest	✅ Champion	0.89	0.87	0.88

(Values shown as example — replace with your real evaluation output from reports/metrics/test_metrics_champion.json.)

🧰 Tech Stack

Python 3.12+

Pandas, NumPy, scikit-learn

Imbalanced-Learn (SMOTE)

XGBoost, LightGBM

Matplotlib, Seaborn

FastAPI, Uvicorn

Joblib, YAML, JSON

📦 Environment Setup
# Create environment
python -m venv winequality
.\winequality\Scripts\activate   # (Windows)
# source winequality/bin/activate  (Linux/Mac)

# Install dependencies
pip install -r requirements.txt


Freeze the exact environment after a successful run:

pip freeze > requirements.lock.txt

🧾 Configuration Example (configs/default.yaml)
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

🧠 Future Work

Add SHAP explainability & feature importance visualizations

Extend FastAPI endpoints to include /metadata and /retrain

Integrate MLflow for experiment management

Deploy on Docker or Azure Container Apps

👩🏽‍💻 Author

Regina Adobea Essien
MSc Data Science Researcher | Ghana Data Science Community
📧 reginaessien83@gmail.com

🔗 LinkedIn
 • GitHub
