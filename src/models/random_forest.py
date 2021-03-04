# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# train random forest model
rf_model = RandomForestRegressor(random_state=1,
                                n_estimators=300)
rf_model.fit(X_train, y_train)
