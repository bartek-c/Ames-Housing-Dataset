# library imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# project imports
from preprocessing import pipeline
from models.xgboost import train_xgboost
from models.random_forest import train_rf
from models.neural_network import train_nn

# load the data
df_train = pd.read_csv("../data/train.csv")
X_test = pd.read_csv("../data/test.csv")
# drop row with missing Electrical value
df_train = df_train.drop(df_train.loc[df_train.Electrical.isnull()].index)
X_train_full = df_train.copy()
y_train_full = X_train_full.pop("SalePrice")
# split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, train_size=0.8, test_size=0.2, random_state=0)

# fit the training examples to the pipeline (excluding cross-val)
pipeline.fit(X_train, y_train)
# transform the datasets
X_train_full_transformed = pipeline.transform(X_train_full)
X_train = pipeline.transform(X_train)
X_val = pipeline.transform(X_val)
X_test = pipeline.transform(X_test)

# train models
xgboost_model = train_xgboost(X_train, y_train, X_val, y_val)
rf_model = train_rf(X_train, y_train)
nn_model = train_nn(X_train, y_train, X_val, y_val)['model']

# dictionary of models
models = [{'model':xgboost_model, 'model_name':'XGBoost'},
          {'model':rf_model, 'model_name':'Random Forest Regressor'},
          {'model':nn_model, 'model_name':'Neural Network'}]

# predict y_test from X_test with a given model and export predictions to csv
def make_predictions(X_test, model, model_name):
    # make predictions on X_test
    preds = model.predict(X_test)
    # create submission dataframe
    sub = pd.read_csv('../data/sample_submission.csv')
    sub['SalePrice'] = preds
    sub = sub.set_index('Id')
    # export to csv
    sub.to_csv(f"../outputs/{model_name}_sub.csv")
    
    return True
  
# make predictions for all models
for model in models:
    print(f"{model['model_name']} - {make_predictions(X_test, model['model'], model['model_name'])}")
