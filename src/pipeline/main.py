"""
Main Pipeline Module - Orchestrates the complete ML pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, Tuple

# Import project modules
from ..data_management import (
    download_wine_dataset, load_wine_data, validate_data_schema,
    clean_data, create_quality_categories, split_data
)
from ..features import (
    create_interaction_features, StandardScalerWrapper,
    select_features_correlation
)
from ..models import (
    ModelTrainer, ModelEvaluator, ModelComparator
)
from ..visualization import create_eda_plots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

 
class WineQualityPipeline:
    """Complete pipeline for wine quality classification."""
    
    def __init__(self, project_root: str = ".", random_state: int = 42):
        self.project_root = Path(project_root)
        self.random_state = random_state
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.trainer = ModelTrainer(random_state=random_state)
        self.evaluator = ModelEvaluator()
        self.comparator = ModelComparator()
        
        # Data storage
        self.raw_data = None
        self.cleaned_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_columns = None
        
    def download_and_load_data(self) -> pd.DataFrame:
        """Download and load the wine quality dataset."""
        logger.info("Starting data download and loading...")
        
        # Download dataset
        raw_file = download_wine_dataset(self.data_dir / "raw")
        
        # Load data
        self.raw_data = load_wine_data(raw_file)
        
        # Validate schema
        is_valid, validation_info = validate_data_schema(self.raw_data)
        if not is_valid:
            logger.warning(f"Data validation issues: {validation_info}")
        
        logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
        return self.raw_data
    
    def preprocess_data(self, 
                       handle_missing: str = 'median',
                       remove_outliers: bool = False,
                       quality_method: str = 'binary',
                       quality_threshold: int = 6) -> pd.DataFrame:
        """Preprocess the dataset."""
        logger.info("Starting data preprocessing...")
        
        # Clean data
        self.cleaned_data = clean_data(
            self.raw_data, 
            handle_missing=handle_missing,
            remove_outliers=remove_outliers
        )
        
        # Create quality categories
        self.cleaned_data = create_quality_categories(
            self.cleaned_data,
            method=quality_method,
            threshold=quality_threshold
        )
        
        # Save processed data
        processed_file = self.data_dir / "processed" / "wine_quality_processed.csv"
        processed_file.parent.mkdir(exist_ok=True)
        self.cleaned_data.to_csv(processed_file, index=False)
        
        logger.info(f"Data preprocessing completed. Shape: {self.cleaned_data.shape}")
        return self.cleaned_data
    
    def create_eda_plots(self) -> Dict[str, str]:
        """Create exploratory data analysis plots."""
        logger.info("Creating EDA plots...")
        
        plot_paths = create_eda_plots(
            self.cleaned_data, 
            str(self.reports_dir),
            target_column='quality'
        )
        
        logger.info(f"EDA plots created: {len(plot_paths)} plots")
        return plot_paths
    
    def engineer_features(self, 
                         create_interactions: bool = True,
                         feature_selection_method: str = 'correlation',
                         n_features: int = 10) -> pd.DataFrame:
        """Engineer features for the model."""
        logger.info("Starting feature engineering...")
        
        # Create interaction features
        if create_interactions:
            self.cleaned_data = create_interaction_features(self.cleaned_data)
            logger.info("Interaction features created")
        
        # Select features
        if feature_selection_method == 'correlation':
            selected_features = select_features_correlation(
                self.cleaned_data, 
                target_column='quality_category',
                threshold=0.1
            )
        else:
            # Use all numeric features except target and quality-related columns
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['quality', 'quality_category', 'quality_label']
            selected_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Store feature columns
        self.feature_columns = selected_features
        
        logger.info(f"Feature engineering completed. Selected {len(selected_features)} features")
        return self.cleaned_data
    
    def split_and_scale_data(self, 
                            test_size: float = 0.2,
                            val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                          pd.Series, pd.Series, pd.Series]:
        """Split data into train/validation/test sets and apply scaling."""
        logger.info("Splitting and scaling data...")
        
        # Split data - use quality_category for binary classification
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            self.cleaned_data,
            target_column='quality_category',
            test_size=test_size,
            val_size=val_size,
            random_state=self.random_state
        )
        
        # Apply scaling
        scaler = StandardScalerWrapper(columns=self.feature_columns)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaled data
        self.train_data = (X_train_scaled, y_train)
        self.val_data = (X_val_scaled, y_val)
        self.test_data = (X_test_scaled, y_test)
        
        # Save scaler
        import joblib
        joblib.dump(scaler, self.models_dir / "scaler.pkl")
        
        logger.info("Data splitting and scaling completed")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_models(self) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and evaluate performance."""
        logger.info("Starting model training...")
        
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Train all models
        training_results = self.trainer.train_all_models(
            X_train, y_train, X_val, y_val
        )
        
        # Evaluate each model
        evaluation_results = {}
        for model_name, results in training_results.items():
            if results.get('training_completed', False):
                # Make predictions
                model = results['model']
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Evaluate performance
                eval_results = self.evaluator.evaluate_classification_performance(
                    y_val, y_pred, y_pred_proba, model_name
                )
                evaluation_results[model_name] = eval_results
                
                # Save model
                model_path = self.models_dir / f"{model_name}.pkl"
                self.trainer.save_model(model_name, str(model_path))
        
        # Compare models
        comparison_df = self.comparator.compare_models(evaluation_results)
        
        # Save results
        self._save_results(evaluation_results, comparison_df)
        
        logger.info("Model training and evaluation completed")
        return evaluation_results
    
    def _save_results(self, 
                     evaluation_results: Dict[str, Dict[str, Any]],
                     comparison_df: pd.DataFrame) -> None:
        """Save evaluation and comparison results."""
        
        # Save evaluation results
        eval_file = self.reports_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for model_name, results in evaluation_results.items():
                json_results[model_name] = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        json_results[model_name][key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_results[model_name][key] = float(value)
                    else:
                        json_results[model_name][key] = value
            json.dump(json_results, f, indent=2)
        
        # Save comparison results
        comparison_file = self.reports_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        logger.info(f"Results saved to {self.reports_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete machine learning pipeline."""
        logger.info("Starting complete wine quality classification pipeline...")
        
        try:
            # 1. Download and load data
            self.download_and_load_data()
            
            # 2. Preprocess data
            self.preprocess_data()
            
            # 3. Create EDA plots
            plot_paths = self.create_eda_plots()
            
            # 4. Engineer features
            self.engineer_features()
            
            # 5. Split and scale data
            self.split_and_scale_data()
            
            # 6. Train models
            evaluation_results = self.train_models()
            
            # 7. Get best model
            best_model_info = self.comparator.get_best_model_info()
            
            logger.info("Complete pipeline finished successfully!")
            
            return {
                'status': 'success',
                'data_shape': self.cleaned_data.shape,
                'feature_count': len(self.feature_columns),
                'plot_paths': plot_paths,
                'evaluation_results': evaluation_results,
                'best_model': best_model_info,
                'models_dir': str(self.models_dir),
                'reports_dir': str(self.reports_dir)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def run_complete_pipeline(project_root: str = ".", random_state: int = 42) -> Dict[str, Any]:
    """Convenience function to run the complete pipeline."""
    pipeline = WineQualityPipeline(project_root, random_state)
    return pipeline.run_complete_pipeline()

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_pipeline()
    print(f"Pipeline completed with status: {results['status']}")