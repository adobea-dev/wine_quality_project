# 🍷 Wine Quality Classification Project

A comprehensive machine learning project that predicts wine quality based on physicochemical properties. This project demonstrates end-to-end data science workflow from data collection to model deployment.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Components](#project-components)
- [Results & Performance](#results--performance)
- [Areas for Improvement](#areas-for-improvement)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project tackles the classic wine quality classification problem using machine learning techniques. The goal is to predict whether a wine is of good quality (score ≥ 6) or bad quality (score < 6) based on various physicochemical properties like acidity, alcohol content, pH levels, and more.

### What You will Learn
- **Data Science Pipeline**: Complete workflow from raw data to deployed models
- **Feature Engineering**: Creating meaningful features from raw data
- **Model Comparison**: Evaluating multiple ML algorithms
- **Data Visualization**: Creating insightful EDA plots
- **Code Organization**: Professional project structure and best practices

## ✨ Features

- 🔄 **Automated Data Pipeline**: Download, preprocess, and validate data
- 📊 **Comprehensive EDA**: 8 different visualization types for data exploration
- 🛠️ **Advanced Feature Engineering**: Interaction features, polynomial features, and feature selection
- 🤖 **Multiple ML Models**: 9 different algorithms for comparison
- 📈 **Model Evaluation**: Detailed performance metrics and comparison
- 💾 **Model Persistence**: Save and load trained models
- 🐳 **Docker Support**: Containerized deployment
- 📱 **Web API**: Flask-based REST API for predictions

## 📁 Project Structure

```
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
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd wine_quality_project
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "from src.pipeline.main import run_complete_pipeline; print('Installation successful!')"
```

## 🏃‍♂️ Quick Start

### Run the Complete Pipeline
```python
from src.pipeline.main import run_complete_pipeline

# Run the entire ML pipeline
results = run_complete_pipeline()
print(f"Pipeline status: {results['status']}")
print(f"Best model: {results['best_model']['model_name']}")
```

### Run Individual Components
```python
from src.pipeline.main import WineQualityPipeline

# Initialize pipeline
pipeline = WineQualityPipeline()

# Step 1: Download and load data
data = pipeline.download_and_load_data()

# Step 2: Preprocess data
processed_data = pipeline.preprocess_data()

# Step 3: Create visualizations
plots = pipeline.create_eda_plots()

# Step 4: Engineer features
pipeline.engineer_features()

# Step 5: Split and scale data
X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_and_scale_data()

# Step 6: Train models
results = pipeline.train_models()
```

## 📚 Detailed Usage

### 1. Data Management

#### Downloading Data
```python
from src.data_management import download_wine_dataset

# Download wine quality dataset
data_file = download_wine_dataset("data/raw")
print(f"Data downloaded to: {data_file}")
```

#### Loading and Validating Data
```python
from src.data_management import load_wine_data, validate_data_schema

# Load data
df = load_wine_data("data/raw/winequalityN.csv")

# Validate data schema
is_valid, validation_info = validate_data_schema(df)
print(f"Data valid: {is_valid}")
```

### 2. Data Preprocessing

```python
from src.data_management import clean_data, create_quality_categories

# Clean the data
cleaned_df = clean_data(df, handle_missing='median', remove_outliers=False)

# Create quality categories (binary classification)
categorized_df = create_quality_categories(cleaned_df, method='binary', threshold=6)
```

### 3. Feature Engineering

```python
from src.features import create_interaction_features, select_features_correlation

# Create interaction features
df_with_interactions = create_interaction_features(df)

# Select best features
selected_features = select_features_correlation(df, target_column='quality_category', threshold=0.1)
```

### 4. Model Training

```python
from src.models import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train all models
results = trainer.train_all_models(X_train, y_train, X_val, y_val)

# Access individual model results
print(f"Logistic Regression score: {results['logistic_regression']['val_score']}")
```

### 5. Model Evaluation

```python
from src.models import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate model performance
eval_results = evaluator.evaluate_classification_performance(y_val, y_pred, y_pred_proba, "model_name")
```

### 6. Visualization

```python
from src.visualization import create_eda_plots

# Create comprehensive EDA plots
plot_paths = create_eda_plots(df, "reports", target_column='quality')
print(f"Created {len(plot_paths)} plots")
```

## 🔧 Project Components

### Data Management Module (`src/data_management/`)

- **`downloader.py`**: Handles data download with Kaggle API and fallback options
- **`loader.py`**: Data loading, validation, and schema checking
- **`processor.py`**: Data cleaning, preprocessing, and train-test splitting

### Features Module (`src/features/`)

- **`engineering.py`**: Creates interaction features, polynomial features, and statistical features
- **`selection.py`**: Implements correlation-based, mutual information, and RFE feature selection
- **`transformation.py`**: Data scaling, normalization, and transformation utilities

### Models Module (`src/models/`)

- **`trainer.py`**: Trains multiple ML models with hyperparameter tuning
- **`evaluator.py`**: Comprehensive model evaluation with multiple metrics
- **`comparator.py`**: Model comparison and ranking functionality

### Visualization Module (`src/visualization/`)

- **`eda_plot.py`**: Creates 8 different types of exploratory data analysis plots
- **`model_plots.py`**: Generates model performance visualizations

## 📊 Results & Performance

### Model Performance Summary
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% |
| Gradient Boosting | 100% | 100% | 100% | 100% |
| SVM | 99.69% | 99.69% | 99.69% | 99.69% |
| KNN | 99.38% | 99.38% | 99.38% | 99.38% |
| Naive Bayes | 100% | 100% | 100% | 100% |
| Decision Tree | 100% | 100% | 100% | 100% |
| AdaBoost | 100% | 100% | 100% | 100% |
| Ridge | 100% | 100% | 100% | 100% |

### Generated Reports
The pipeline automatically generates:
- **8 EDA Plots**: Data overview, target distribution, feature distributions, correlation heatmap, box plots, pair plots, missing values analysis, and outlier analysis
- **Model Comparison CSV**: Detailed performance metrics for all models
- **Evaluation Results JSON**: Complete evaluation results with confusion matrices and ROC curves

## 🚀 Web API Usage

### Start the Flask Application
```bash
python app.py
```

### Make Predictions via API
```bash
# Example prediction request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

## 🐳 Docker Deployment

### Build and Run with Docker
```bash
# Build the Docker image
docker build -t wine-quality-project .

# Run the container
docker run -p 5000:5000 wine-quality-project
```

### Using Docker Compose
```bash
# Start the application
docker-compose up

# Run in background
docker-compose up -d
```

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run Tests with Coverage
```bash
pytest --cov=src tests/
```

### Run Specific Test Module
```bash
pytest tests/test_models.py
```

## 📈 Areas for Improvement

### 1. **Data Quality & Collection**
- **Larger Dataset**: Collect more wine samples to improve model generalization
- **Feature Engineering**: Add domain-specific features like wine aging, region, grape variety
- **Data Augmentation**: Implement synthetic data generation techniques
- **External Data**: Incorporate weather data, soil composition, or winery information

### 2. **Model Performance**
- **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
- **Hyperparameter Tuning**: Add automated hyperparameter optimization using Optuna or similar
- **Ensemble Methods**: Implement voting classifiers or stacking ensembles
- **Deep Learning**: Experiment with neural networks for complex pattern recognition

### 3. **Feature Engineering**
- **Advanced Features**: Create more sophisticated interaction terms and polynomial features
- **Feature Selection**: Implement more advanced selection methods (LASSO, Elastic Net)
- **Dimensionality Reduction**: Add PCA, t-SNE, or UMAP for visualization and feature reduction
- **Time Series Features**: If temporal data is available, add time-based features

### 4. **Model Interpretability**
- **SHAP Values**: Add SHAP (SHapley Additive exPlanations) for model interpretability
- **Feature Importance**: Implement permutation importance and partial dependence plots
- **LIME**: Add Local Interpretable Model-agnostic Explanations
- **Model Cards**: Create detailed model documentation

### 5. **Production Readiness**
- **API Documentation**: Add Swagger/OpenAPI documentation
- **Logging**: Implement comprehensive logging with different levels
- **Monitoring**: Add model performance monitoring and drift detection
- **A/B Testing**: Implement framework for model comparison in production

### 6. **Code Quality**
- **Type Hints**: Add comprehensive type hints throughout the codebase
- **Documentation**: Add docstrings and inline comments
- **Error Handling**: Implement robust error handling and recovery mechanisms
- **Configuration Management**: Use YAML or JSON config files for parameters

### 7. **Scalability**
- **Distributed Training**: Implement distributed model training for large datasets
- **Batch Processing**: Add batch prediction capabilities
- **Caching**: Implement caching for frequently used data and models
- **Database Integration**: Add database support for data storage and retrieval

### 8. **User Experience**
- **Interactive Dashboard**: Create a web-based dashboard for data exploration
- **Real-time Predictions**: Implement streaming predictions
- **Mobile App**: Develop a mobile application for wine quality prediction
- **Visualization Improvements**: Add interactive plots with Plotly or Bokeh

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new functionality
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Issues**: Look through existing GitHub issues
2. **Create a New Issue**: Describe your problem with details
3. **Contact**: Reach out to the project maintainers

## 🙏 Acknowledgments

- **Dataset**: Wine Quality Dataset from UCI Machine Learning Repository
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn, and the open-source community
- **Inspiration**: This project was inspired by the need to understand wine quality prediction using machine learning

---

**Happy Learning ACITY! 🍷📊🤖**

*This project is designed for educational purposes and demonstrates best practices in machine learning project development. Feel free to use it as a template for your own projects!*
