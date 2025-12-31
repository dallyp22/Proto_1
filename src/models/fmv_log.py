"""
FMV Model with Log-Price transformation.
Often performs better on skewed price distributions.

Core valuation models and methodologies are licensed from Dallas Polivka.
Copyright (c) 2025 Dallas Polivka
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from src.features.pipeline import FeaturePipeline
from .config import FMV_PARAMS, TRAINING_CONFIG


class FMVLogModel:
    """Fair Market Value prediction model using log-transformed prices."""
    
    def __init__(self, params: Optional[Dict] = None):
        self.params = {**FMV_PARAMS, **(params or {})}
        self.model: Optional[lgb.Booster] = None
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.feature_columns: List[str] = []
        self.categorical_features: List[str] = []
        self.metadata: Dict = {}
    
    def train(
        self,
        df: pd.DataFrame,
        feature_pipeline: Optional[FeaturePipeline] = None,
        target_col: str = 'price',
        val_fraction: float = TRAINING_CONFIG['val_fraction'],
        test_fraction: float = TRAINING_CONFIG['test_fraction'],
        n_estimators: int = TRAINING_CONFIG['n_estimators'],
        early_stopping_rounds: int = TRAINING_CONFIG['early_stopping_rounds'],
    ) -> Dict:
        """Train the model with log-transformed prices."""
        
        print("=" * 60)
        print("TRAINING FMV MODEL (LOG-PRICE)")
        print("=" * 60)
        
        # Sort by date
        df = df.sort_values('sold_date').reset_index(drop=True)
        
        # Time-based splits
        n = len(df)
        train_end = int(n * (1 - val_fraction - test_fraction))
        val_end = int(n * (1 - test_fraction))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"\nData Splits:")
        print(f"  Train: {len(train_df):,} ({train_df['sold_date'].min().date()} to {train_df['sold_date'].max().date()})")
        print(f"  Val:   {len(val_df):,} ({val_df['sold_date'].min().date()} to {val_df['sold_date'].max().date()})")
        print(f"  Test:  {len(test_df):,} ({test_df['sold_date'].min().date()} to {test_df['sold_date'].max().date()})")
        
        # Feature engineering
        print("\nFeature Engineering...")
        if feature_pipeline is None:
            self.feature_pipeline = FeaturePipeline()
            train_features = self.feature_pipeline.fit_transform(train_df)
        else:
            self.feature_pipeline = feature_pipeline
            train_features = self.feature_pipeline.transform(train_df)
        
        val_features = self.feature_pipeline.transform(val_df)
        test_features = self.feature_pipeline.transform(test_df)
        
        # Get feature columns
        cat_cols, num_cols = self.feature_pipeline.get_feature_columns()
        self.categorical_features = [c for c in cat_cols if c in train_features.columns]
        numeric_features = [c for c in num_cols if c in train_features.columns]
        self.feature_columns = self.categorical_features + numeric_features
        
        print(f"Features: {len(self.feature_columns)} ({len(self.categorical_features)} categorical, {len(numeric_features)} numeric)")
        
        # Prepare data
        X_train = self._prepare_features(train_features)
        X_val = self._prepare_features(val_features)
        X_test = self._prepare_features(test_features)
        
        # LOG TRANSFORM THE TARGET
        y_train_log = np.log(train_df[target_col])
        y_val_log = np.log(val_df[target_col])
        y_test_log = np.log(test_df[target_col])
        
        print("\n📊 Target transformation: price → log(price)")
        print(f"  Original price range: ${train_df[target_col].min():,.0f} - ${train_df[target_col].max():,.0f}")
        print(f"  Log-price range: {y_train_log.min():.2f} - {y_train_log.max():.2f}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train_log, categorical_feature=self.categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val_log, reference=train_data)
        
        # Train
        print("\nTraining...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(TRAINING_CONFIG['verbose_eval']),
            ],
        )
        
        # Evaluate (convert back to original scale)
        metrics = {}
        for name, X, y_log, y_actual in [
            ('val', X_val, y_val_log, val_df[target_col]), 
            ('test', X_test, y_test_log, test_df[target_col])
        ]:
            # Predict in log space
            preds_log = self.model.predict(X)
            # Transform back to price space
            preds = np.exp(preds_log)
            
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_actual, preds))
            metrics[f'{name}_mape'] = mean_absolute_percentage_error(y_actual, preds) * 100
            metrics[f'{name}_r2'] = r2_score(y_actual, preds)
        
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'model_type': 'log_price',
            'n_train': len(train_df),
            'n_val': len(val_df),
            'n_test': len(test_df),
            'best_iteration': self.model.best_iteration,
            'metrics': metrics,
        }
        
        # Results
        print("\n" + "=" * 60)
        print("RESULTS (LOG-PRICE MODEL)")
        print("=" * 60)
        print(f"\nValidation:  RMSE=${metrics['val_rmse']:,.0f}  MAPE={metrics['val_mape']:.1f}%  R²={metrics['val_r2']:.3f}")
        print(f"Test:        RMSE=${metrics['test_rmse']:,.0f}  MAPE={metrics['test_mape']:.1f}%  R²={metrics['test_r2']:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions (automatically converts from log to price)."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        features = self.feature_pipeline.transform(df)
        X = self._prepare_features(features)
        
        # Predict in log space, convert back to price
        preds_log = self.model.predict(X)
        return np.exp(preds_log)
    
    def feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance,
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path / 'model.lgb'))
        self.feature_pipeline.save(str(path / 'pipeline.joblib'))
        
        with open(path / 'metadata.json', 'w') as f:
            json.dump({
                'model_type': 'log_price',
                'feature_columns': self.feature_columns,
                'categorical_features': self.categorical_features,
                'metadata': self.metadata,
                'params': self.params,
            }, f, indent=2, default=str)
        
        print(f"Log-price model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FMVLogModel':
        """Load model."""
        path = Path(path)
        
        instance = cls()
        instance.model = lgb.Booster(model_file=str(path / 'model.lgb'))
        instance.feature_pipeline = FeaturePipeline.load(str(path / 'pipeline.joblib'))
        
        with open(path / 'metadata.json') as f:
            data = json.load(f)
        
        instance.feature_columns = data['feature_columns']
        instance.categorical_features = data['categorical_features']
        instance.metadata = data['metadata']
        instance.params = data['params']
        
        return instance
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model."""
        cols = [c for c in self.feature_columns if c in df.columns]
        X = df[cols].copy()
        
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        return X

