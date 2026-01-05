"""
Smart Model Router - Selects best available model with fallback logic.

Priority:
1. Make-Category model (if available) - Most specific
2. Category model (if available) - General category
3. Generic model - Fallback
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from .fmv_log import FMVLogModel
from .make_category_config import get_make_category_key, CATEGORY_FILTERS
from .multi_model_config import get_category_key


class SmartModelRouter:
    """
    Intelligently routes predictions to the best available model.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._cache = {}
    
    def get_best_model(
        self, 
        make: str, 
        category: str, 
        method: str = 'log'
    ) -> Tuple[Optional[FMVLogModel], str]:
        """
        Get the best available model for given make and category.
        
        Returns:
            (model, model_type) where model_type is:
            - 'make_category' (most specific)
            - 'category' (general)
            - 'generic' (fallback)
            - None (no model found)
        """
        suffix = '_log' if method == 'log' else ''
        
        # Try 1: Make-Category specific model (BEST)
        make_cat_key = get_make_category_key(make, category)
        if make_cat_key:
            model_path = self.models_dir / f"fmv_{make_cat_key}{suffix}"
            if model_path.exists():
                cache_key = f"{make_cat_key}{suffix}"
                if cache_key not in self._cache:
                    try:
                        self._cache[cache_key] = FMVLogModel.load(str(model_path))
                    except:
                        pass
                if cache_key in self._cache:
                    return self._cache[cache_key], 'make_category'
        
        # Try 2: Category model (GOOD)
        cat_key = get_category_key(category)
        model_path = self.models_dir / f"fmv_{cat_key}{suffix}"
        if model_path.exists():
            cache_key = f"{cat_key}{suffix}"
            if cache_key not in self._cache:
                try:
                    self._cache[cache_key] = FMVLogModel.load(str(model_path))
                except:
                    pass
            if cache_key in self._cache:
                return self._cache[cache_key], 'category'
        
        # Try 3: Generic model (FALLBACK)
        model_path = self.models_dir / f"fmv_other{suffix}"
        if model_path.exists():
            cache_key = f"other{suffix}"
            if cache_key not in self._cache:
                try:
                    self._cache[cache_key] = FMVLogModel.load(str(model_path))
                except:
                    pass
            if cache_key in self._cache:
                return self._cache[cache_key], 'generic'
        
        return None, None
    
    def predict(
        self, 
        equipment_data: pd.DataFrame,
        make: str,
        category: str,
        method: str = 'log'
    ) -> Tuple[float, str, dict]:
        """
        Make prediction using best available model.
        
        Returns:
            (prediction, model_type, model_info)
        """
        model, model_type = self.get_best_model(make, category, method)
        
        if model is None:
            raise ValueError(f"No model found for {make} - {category}")
        
        prediction = model.predict(equipment_data)[0]
        
        model_info = {
            'model_type': model_type,
            'test_mape': model.metadata.get('metrics', {}).get('test_mape'),
            'test_r2': model.metadata.get('metrics', {}).get('test_r2'),
            'n_train': model.metadata.get('n_train'),
        }
        
        return prediction, model_type, model_info
    
    def list_available_models(self) -> dict:
        """List all available models by type."""
        available = {
            'make_category': [],
            'category': [],
            'generic': [],
        }
        
        # Check make-category models
        for key in MAKE_CATEGORY_MODELS.keys():
            for suffix in ['_log', '']:
                path = self.models_dir / f"fmv_{key}{suffix}"
                if path.exists():
                    available['make_category'].append(f"{key}{suffix}")
        
        # Check category models
        from .multi_model_config import CATEGORY_MODELS
        for key in CATEGORY_MODELS.keys():
            for suffix in ['_log', '']:
                path = self.models_dir / f"fmv_{key}{suffix}"
                if path.exists():
                    available['category'].append(f"{key}{suffix}")
        
        # Check generic
        for suffix in ['_log', '']:
            path = self.models_dir / f"fmv_other{suffix}"
            if path.exists():
                available['generic'].append(f"other{suffix}")
        
        return available
