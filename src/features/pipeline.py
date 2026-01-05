"""
Feature engineering pipeline with actual schema column names.

Core valuation models and methodologies are licensed from Dallas Polivka.
Copyright (c) 2025 Dallas Polivka
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
        'make_model_key',  # Combined make-model (e.g., "john-deere-8320r")
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
        'condition_score',   # Numeric condition rating (1-5)
        'horsepower',        # Engine power - CRITICAL price driver
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
        
        # Make-Model volumes (combined popularity)
        if 'make_model_key' in df.columns:
            self._make_volumes = df['make_model_key'].value_counts().to_dict()
        elif 'make_key' in df.columns:
            self._make_volumes = df['make_key'].value_counts().to_dict()
        
        # Keep model volumes for backward compatibility
        if 'raw_model' in df.columns:
            self._model_volumes = df['raw_model'].value_counts().to_dict()
        
        # Valid categories (for rare handling)
        for col in ['make_model_key', 'region', 'raw_category']:
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
    
    def _parse_specs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse equipment specifications from JSON field."""
        import json
        
        if 'specs' not in df.columns:
            # If no specs field, set defaults
            df['horsepower'] = 150.0  # Default HP
            return df
        
        # Parse JSON specs (convert Ruby hash notation to JSON)
        def safe_parse_specs(spec_str):
            if pd.isna(spec_str) or not spec_str:
                return {}
            try:
                # Convert Ruby notation to JSON
                json_str = str(spec_str).replace('=>', ':').replace('nil', 'null')
                return json.loads(json_str)
            except:
                return {}
        
        df['specs_dict'] = df['specs'].apply(safe_parse_specs)
        
        # Extract horsepower - CRITICAL price driver!
        df['horsepower'] = df['specs_dict'].apply(
            lambda x: x.get('horsepower') if isinstance(x, dict) else None
        )
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        
        # Impute missing horsepower by make_model_key
        if 'make_model_key' in df.columns and df['horsepower'].notna().any():
            df['horsepower'] = df['horsepower'].fillna(
                df.groupby('make_model_key')['horsepower'].transform('median')
            )
        
        # Final fallback to category median or default
        if df['horsepower'].isna().any():
            if 'raw_category' in df.columns:
                df['horsepower'] = df['horsepower'].fillna(
                    df.groupby('raw_category')['horsepower'].transform('median')
                )
            df['horsepower'] = df['horsepower'].fillna(150.0)  # Overall default
        
        return df
    
    def _add_equipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Equipment-derived features."""
        
        # Parse specifications from JSON
        df = self._parse_specs(df)
        
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
            
            # Add NUMERIC condition score (1-5 scale)
            condition_scores = {
                'excellent': 5.0,
                'good': 4.0,
                'fair': 3.0,
                'poor': 2.0,
            }
            df['condition_score'] = df['condition'].map(condition_scores).fillna(4.0)
        else:
            df['condition'] = 'good'  # Default if not available
            df['condition_score'] = 4.0
        
        # Equipment age
        if 'sold_date' in df.columns and 'year' in df.columns:
            sale_year = pd.to_datetime(df['sold_date']).dt.year
            df['equipment_age'] = (sale_year - df['year']).clip(lower=0, upper=self.config.equipment_age_cap)
        
        # Hours per year (use pre-calculated if available, otherwise calculate)
        if 'hours_per_year' not in df.columns or df['hours_per_year'].isna().all():
            if 'hours' in df.columns and 'equipment_age' in df.columns:
                df['hours_per_year'] = np.where(
                    df['equipment_age'] > 0,
                    df['hours'] / df['equipment_age'],
                    df['hours']
                )
        
        # Clip to reasonable max
        if 'hours_per_year' in df.columns:
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
        
        # Make-Model volume (combined popularity/desirability)
        if 'make_model_key' in df.columns:
            df['make_volume'] = df['make_model_key'].map(self._make_volumes).fillna(1)
            df['log_make_volume'] = np.log1p(df['make_volume'])
        elif 'make_key' in df.columns:
            df['make_volume'] = df['make_key'].map(self._make_volumes).fillna(1)
            df['log_make_volume'] = np.log1p(df['make_volume'])
        
        # Model volume (for backward compatibility or if separate model field used)
        if 'raw_model' in df.columns and self._model_volumes:
            df['model_volume'] = df['raw_model'].map(self._model_volumes).fillna(1)
            df['log_model_volume'] = np.log1p(df['model_volume'])
        else:
            df['model_volume'] = 1
            df['log_model_volume'] = 0
        
        return df
    
    def _handle_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map rare categories to 'other'."""
        
        for col in ['make_model_key', 'region', 'raw_category']:
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

