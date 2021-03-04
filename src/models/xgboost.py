# imports
from xgboost import XGBRegressor

# train XGBoost model
def train_xgboost(X_train, y_train, X_val, y_val):
  xgboost_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
  xgboost_model.fit(X_train, y_train, 
               early_stopping_rounds=5, 
               eval_set=[(X_val, y_val)], 
               verbose=False)
  return xgboost_model
