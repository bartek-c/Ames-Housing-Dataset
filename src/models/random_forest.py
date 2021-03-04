# imports
from sklearn.ensemble import RandomForestRegressor

# train random forest model
def train_rf(X_train, y_train):
  rf_model = RandomForestRegressor(random_state=1,
                                  n_estimators=300)
  rf_model.fit(X_train, y_train)
  
  return rf_model
