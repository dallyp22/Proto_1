"""
Model configuration.
"""

FMV_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': -1,
    'min_data_in_leaf': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42,
}

TRAINING_CONFIG = {
    'n_estimators': 2000,
    'early_stopping_rounds': 50,
    'verbose_eval': 100,
    'val_fraction': 0.15,
    'test_fraction': 0.15,
}

