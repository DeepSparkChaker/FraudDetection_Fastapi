# -*- coding: utf-8 -*-
"""Model config in json format"""
cfg = {
    "data": {
        "path": "C:/Users/rzouga/Desktop/ALLINHERE/ALLINHERE/FraudDetection/transactions_train.csv"
    },
    # "data_test": {
    #   "path": "../input/ventilator-pressure-prediction/test.csv"
    # },
    # "data_submission": {
    #   "path": "../input/ventilator-pressure-prediction/test.csv"
    # },
    "train": {
        'fit_params': {'early_stopping_rounds': 100, 'verbose': 55000},
        'n_fold': 5,
        'seeds': [2021],
        'target_col': "pressure",
        'debug': False

    },
    "model": {
        "learning_rate": 0.008,
        'metric': 'auc',
        'objective': 'binary',
        'device': 'gpu',
        'n_estimators': 3205,
        'num_leaves': 184,
        'min_child_samples': 63,
        'feature_fraction': 0.6864594334728974,
        'bagging_fraction': 0.9497327922401265,
        'bagging_freq': 1,
        'reg_alpha': 19,
        'reg_lambda': 19,
        'gpu_platform_id': 0
    }
}