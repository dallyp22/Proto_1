"""
Feature engineering pipeline with actual schema column names.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import joblib


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    hours_per_year_cap: float = 2000
    equipment_age_cap: int = 50
    min_category_frequency: int = 20


class FeaturePipeline:
    """
    Feature engineering pipeline.
    """
    
    CATEGORICAL_FEATURES = [
        'make_key',
        'region',
        'utilization_bucket',
        'raw_category',
        'condition',  # CRITICAL: Excellent vs Poor = huge price difference
    ]
    
    NUMERIC_FEATURES = [
        'year',
        'hours',
        'equipment_age',
        'hours_per_year',
        'is_current_production',
        'years_since_discontinued',
        'sale_month',
        'sale_quarter',
        'month_sin',
        'month_cos',
        'is_planting_season',
        'is_harvest_season',
        'barometer_norm',
        'sentiment_spread',
        'investment_confidence',
        'diesel_relative',
        'el_nino_phase',
        'log_make_volume',
        'log_model_volume',  # Model popularity/desirability
    ]
    
    TARGET = 'price'
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._fitted = False
        self._reference_values: Dict = {}
        self._make_volumes: Dict = {}
        self._model_volumes: Dict = {}
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """Learn reference values from training data."""
        
        # Barometer stats
        if 'barometer' in df.columns:
            self._reference_values['barometer_mean'] = df['barometer'].mean()
            self._reference_values['barometer_std'] = df['barometer'].std()
        
        # Diesel stats
        if 'diesel_price' in df.columns:
            self._reference_values['diesel_mean'] = df['diesel_price'].mean()
        
        # Make volumes
        if 'make_key' in df.columns:
            self._make_volumes = df['make_key'].value_counts().to_dict()
        
        # Model volumes (model popularity)
        if 'raw_model' in df.columns:
            self._model_volumes = df['raw_model'].value_counts().to_dict()
        
        # Valid categories (for rare handling)
        for col in ['make_key', 'region', 'raw_category']:
            if col in df.columns:
                counts = df[col].value_counts()
                valid = counts[counts >= self.config.min_category_frequency].index
                self._reference_values[f'{col}_valid'] = set(valid)
        
        self._fitted = True
        print("Pipeline fitted")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        if not self._fitted:
            raise ValueError("Pipeline must be fit first")
        
        df = df.copy()
        
        df = self._add_equipment_features(df)
        df = self._add_temporal_features(df)
        df = self._add_macro_features(df)
        df = self._add_density_features(df)
        df = self._handle_rare_categories(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    
    def get_feature_columns(self) -> Tuple[List[str], List[str]]:
        """Return (categorical, numeric) feature lists."""
        return self.CATEGORICAL_FEATURES.copy(), self.NUMERIC_FEATURES.copy()
    
    def _add_equipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Equipment-derived features."""
        
        # Parse and standardize condition
        if 'raw_condition' in df.columns:
            df['condition'] = df['raw_condition'].astype(str).str.lower().str.strip()
            # Standardize common variations
            condition_map = {
                'excellent': 'excellent',
                'good': 'good',
                'fair': 'fair',
                'poor': 'poor',
                'average': 'fair',
                'very good': 'good',
                'like new': 'excellent',
                'needs work': 'poor',
                'salvage': 'poor',
            }
            df['condition'] = df['condition'].map(lambda x: condition_map.get(x, 'good'))
            df['condition'] = df['condition'].fillna('good')  # Default to good
        else:
            df['condition'] = 'good'  # Default if not available
        
        # Equipment age
        if 'sold_date' in df.columns and 'year' in df.columns:
            sale_year = pd.to_datetime(df['sold_date']).dt.year
            df['equipment_age'] = (sale_year - df['year']).clip(lower=0, upper=self.config.equipment_age_cap)
        
        # Hours per year
        if 'hours' in df.columns and 'equipment_age' in df.columns:
            df['hours_per_year'] = np.where(
                df['equipment_age'] > 0,
                df['hours'] / df['equipment_age'],
                df['hours']
            )
            df['hours_per_year'] = df['hours_per_year'].clip(upper=self.config.hours_per_year_cap)
            
            # Utilization bucket
            df['utilization_bucket'] = pd.cut(
                df['hours_per_year'].fillna(500),
                bins=[0, 300, 600, 1000, float('inf')],
                labels=['light', 'normal', 'heavy', 'extreme']
            )
        
        # Production status (from make data)
        if 'make_production_year_end' in df.columns:
            df['is_current_production'] = df['make_production_year_end'].isna().astype(int)
            
            if 'sold_date' in df.columns:
                sale_year = pd.to_datetime(df['sold_date']).dt.year
                df['years_since_discontinued'] = np.where(
                    df['make_production_year_end'].notna(),
                    (sale_year - df['make_production_year_end']).clip(lower=0),
                    0
                ).astype(float)  # Ensure numeric type
        else:
            df['is_current_production'] = 1
            df['years_since_discontinued'] = 0.0
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features."""
        
        if 'sold_date' not in df.columns:
            return df
        
        sold_date = pd.to_datetime(df['sold_date'])
        
        df['sale_month'] = sold_date.dt.month
        df['sale_quarter'] = sold_date.dt.quarter
        df['sale_year'] = sold_date.dt.year
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12)
        
        # Agricultural seasons
        df['is_planting_season'] = df['sale_month'].isin([2, 3, 4, 5]).astype(int)
        df['is_harvest_season'] = df['sale_month'].isin([8, 9, 10, 11]).astype(int)
        
        return df
    
    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Economic indicator features."""
        
        # Barometer normalization
        if 'barometer' in df.columns:
            mean = self._reference_values.get('barometer_mean', 100)
            std = self._reference_values.get('barometer_std', 20)
            if std > 0:
                df['barometer_norm'] = (df['barometer'] - mean) / std
            else:
                df['barometer_norm'] = 0
        
        # Sentiment spread
        if 'future_expectations' in df.columns and 'current_conditions' in df.columns:
            df['sentiment_spread'] = df['future_expectations'] - df['current_conditions']
        
        # Investment confidence
        if 'capital_investment_index' in df.columns:
            df['investment_confidence'] = df['capital_investment_index'] / 100
        
        # Diesel relative
        if 'diesel_price' in df.columns:
            mean = self._reference_values.get('diesel_mean', 3.5)
            if mean > 0:
                df['diesel_relative'] = df['diesel_price'] / mean
        
        return df
    
    def _add_density_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data density features."""
        
        # Make volume
        if 'make_key' in df.columns:
            df['make_volume'] = df['make_key'].map(self._make_volumes).fillna(1)
            df['log_make_volume'] = np.log1p(df['make_volume'])
        
        # Model volume (popularity/desirability indicator)
        if 'raw_model' in df.columns:
            df['model_volume'] = df['raw_model'].map(self._model_volumes).fillna(1)
            df['log_model_volume'] = np.log1p(df['model_volume'])
        else:
            df['model_volume'] = 1
            df['log_model_volume'] = 0
        
        return df
    
    def _handle_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map rare categories to 'other'."""
        
        for col in ['make_key', 'region', 'raw_category']:
            if col in df.columns:
                valid_key = f'{col}_valid'
                if valid_key in self._reference_values:
                    valid = self._reference_values[valid_key]
                    df[col] = df[col].apply(lambda x: x if x in valid else 'other')
        
        return df
    
    def save(self, path: str):
        """Save fitted pipeline."""
        joblib.dump({
            'config': self.config,
            'reference_values': self._reference_values,
            'make_volumes': self._make_volumes,
            'model_volumes': self._model_volumes,
            'fitted': self._fitted,
        }, path)
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        """Load saved pipeline."""
        data = joblib.load(path)
        instance = cls(config=data['config'])
        instance._reference_values = data['reference_values']
        instance._make_volumes = data['make_volumes']
        instance._model_volumes = data.get('model_volumes', {})  # Backward compatible
        instance._fitted = data['fitted']
        return instance

