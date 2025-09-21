"""
Data Preprocessor Module

Handles data cleaning, preprocessing, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame, 
               handle_missing: str = 'median',
               remove_outliers: bool = False,
               outlier_method: str = 'iqr') -> pd.DataFrame:
    """
    Clean the Wine Quality dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        handle_missing (str): Method to handle missing values ('median', 'mean', 'drop')
        remove_outliers (bool): Whether to remove outliers
        outlier_method (str): Method to detect outliers ('iqr', 'zscore')
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    logger.info("Starting data cleaning process...")
    df_clean = df.copy()
    
    # Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    logger.info(f"Missing values before cleaning: {missing_before}")
    
    if missing_before > 0:
        if handle_missing == 'drop':
            df_clean = df_clean.dropna()
            logger.info("Dropped rows with missing values")
        elif handle_missing == 'median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            logger.info("Filled missing values with median")
        elif handle_missing == 'mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
            logger.info("Filled missing values with mean")
    
    # Handle categorical missing values
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_value)
            logger.info(f"Filled missing values in {col} with mode: {mode_value}")
    
    # Remove outliers if requested
    if remove_outliers:
        outliers_removed = 0
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if outlier_method == 'iqr':
            for col in numeric_cols:
                if col != 'quality':  # Don't remove outliers from target
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    before_count = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    outliers_removed += before_count - len(df_clean)
            
            logger.info(f"Removed {outliers_removed} outlier rows using IQR method")
        
        elif outlier_method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                if col != 'quality':  # Don't remove outliers from target
                    z_scores = np.abs(stats.zscore(df_clean[col]))
                    before_count = len(df_clean)
                    df_clean = df_clean[z_scores < 3]
                    outliers_removed += before_count - len(df_clean)
            
            logger.info(f"Removed {outliers_removed} outlier rows using Z-score method")
    
    # Encode categorical variables
    if 'type' in df_clean.columns:
        le = LabelEncoder()
        df_clean['type_encoded'] = le.fit_transform(df_clean['type'])
        logger.info(f"Encoded 'type' column: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    missing_after = df_clean.isnull().sum().sum()
    logger.info(f"Missing values after cleaning: {missing_after}")
    logger.info(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean

def create_quality_categories(df: pd.DataFrame, 
                            method: str = 'binary',
                            threshold: int = 6) -> pd.DataFrame:
    """
    Create quality categories from numeric quality scores.
    
    Args:
        df (pd.DataFrame): Dataset with quality column
        method (str): Categorization method ('binary', 'multi', 'custom')
        threshold (int): Threshold for binary classification
        
    Returns:
        pd.DataFrame: Dataset with quality categories
    """
    df_cat = df.copy()
    
    if method == 'binary':
        # Binary classification: Good (>=threshold) vs Bad (<threshold)
        df_cat['quality_category'] = (df_cat['quality'] >= threshold).astype(int)
        df_cat['quality_label'] = df_cat['quality_category'].map({0: 'Bad', 1: 'Good'})
        logger.info(f"Created binary quality categories (threshold={threshold})")
        
    elif method == 'multi':
        # Multi-class: Low (3-4), Medium (5-6), High (7-9)
        def categorize_quality(score):
            if score <= 4:
                return 0  # Low
            elif score <= 6:
                return 1  # Medium
            else:
                return 2  # High
        
        df_cat['quality_category'] = df_cat['quality'].apply(categorize_quality)
        df_cat['quality_label'] = df_cat['quality_category'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        logger.info("Created multi-class quality categories (Low/Medium/High)")
        
    elif method == 'custom':
        # Custom: Very Low (3), Low (4), Medium (5-6), High (7-8), Very High (9)
        def categorize_quality_custom(score):
            if score == 3:
                return 0  # Very Low
            elif score == 4:
                return 1  # Low
            elif score in [5, 6]:
                return 2  # Medium
            elif score in [7, 8]:
                return 3  # High
            else:  # score == 9
                return 4  # Very High
        
        df_cat['quality_category'] = df_cat['quality'].apply(categorize_quality_custom)
        df_cat['quality_label'] = df_cat['quality_category'].map({
            0: 'Very Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        })
        logger.info("Created custom quality categories (5 classes)")
    
    return df_cat

def split_data(df: pd.DataFrame, 
               target_column: str = 'quality',
               test_size: float = 0.2,
               val_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                             pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Dataset to split
        target_column (str): Name of target column
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of remaining data for validation set
        random_state (int): Random seed for reproducibility
        stratify (bool): Whether to stratify based on target variable
        
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data into train/validation/test sets...")
    
    # Separate features and target
    # Also exclude other quality-related columns
    exclude_cols = [target_column, 'quality', 'quality_label']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df[target_column]
    
    # First split: train+val vs test
    stratify_param = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    # Second split: train vs val
    if stratify:
        stratify_param = y_temp
    else:
        stratify_param = None
    
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=stratify_param
    )
    
    logger.info(f"Data split completed:")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Validation: {X_val.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_feature_importance_correlation(df: pd.DataFrame, 
                                     target_column: str = 'quality') -> pd.DataFrame:
    """
    Calculate correlation between features and target variable.
    
    Args:
        df (pd.DataFrame): Dataset
        target_column (str): Name of target column
        
    Returns:
        pd.DataFrame: Correlation matrix with target variable
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    correlations = df[numeric_cols + [target_column]].corr()[target_column].abs().sort_values(ascending=False)
    
    return correlations.drop(target_column)  # Remove self-correlation