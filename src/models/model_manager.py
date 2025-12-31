"""
Model Manager for handling multiple category-specific FMV models.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

from .fmv import FMVModel
from .multi_model_config import (
    CATEGORY_MODELS, 
    GENERIC_MODEL,
    get_category_key,
    get_all_model_keys,
    get_model_display_name
)


class ModelManager:
    """
    Manages multiple category-specific FMV models.
    Automatically selects and uses the appropriate model based on equipment category.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._models: Dict[str, FMVModel] = {}
        self._loaded_categories = set()
    
    def load_model(self, category_key: str) -> FMVModel:
        """Load a specific category model."""
        if category_key in self._models:
            return self._models[category_key]
        
        model_path = self.models_dir / f"fmv_{category_key}"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found for category '{category_key}' at {model_path}\n"
                f"Please train the model first using train_all_models.py"
            )
        
        model = FMVModel.load(str(model_path))
        self._models[category_key] = model
        self._loaded_categories.add(category_key)
        
        return model
    
    def load_all_models(self):
        """Load all available category models."""
        for category_key in get_all_model_keys():
            try:
                self.load_model(category_key)
                print(f"✓ Loaded {get_model_display_name(category_key)} model")
            except FileNotFoundError:
                print(f"✗ Model not found: {category_key}")
    
    def predict(self, df: pd.DataFrame, category: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using the appropriate category model.
        
        Args:
            df: Input dataframe with equipment details
            category: Category key or raw_category value. If None, will try to detect from df.
        
        Returns:
            Array of predicted prices
        """
        # Determine category
        if category is None:
            if 'raw_category' not in df.columns:
                raise ValueError("Must provide category or include 'raw_category' in dataframe")
            category = df['raw_category'].iloc[0]
        
        # Map to model key
        model_key = get_category_key(category)
        
        # Load and use appropriate model
        model = self.load_model(model_key)
        return model.predict(df)
    
    def get_model_info(self, category_key: str) -> Dict:
        """Get information about a specific model."""
        model = self.load_model(category_key)
        
        return {
            'category': get_model_display_name(category_key),
            'category_key': category_key,
            'trained_at': model.metadata.get('trained_at'),
            'n_train': model.metadata.get('n_train'),
            'test_mape': model.metadata.get('metrics', {}).get('test_mape'),
            'test_r2': model.metadata.get('metrics', {}).get('test_r2'),
            'test_rmse': model.metadata.get('metrics', {}).get('test_rmse'),
        }
    
    def list_available_models(self) -> list:
        """List all available trained models."""
        available = []
        
        for category_key in get_all_model_keys():
            model_path = self.models_dir / f"fmv_{category_key}"
            if model_path.exists():
                try:
                    info = self.get_model_info(category_key)
                    available.append(info)
                except Exception:
                    pass
        
        return available
    
    def get_feature_importance(self, category_key: str) -> pd.DataFrame:
        """Get feature importance for a specific category model."""
        model = self.load_model(category_key)
        return model.feature_importance()

