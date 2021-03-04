# imports
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# train XGBoost model
xgboost_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgboost_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_val, y_val)], 
             verbose=False)
