# library imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import MEstimateEncoder

# project imports
from categorical_features import cat_features

# drop cols and create features from aggregated bathrooms and porch areas
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to drop given columns; aggregate bathrooms (TotBaths); sum porch areas (PorchArea)
    and use k-means to group data into clusters (YrCluster).
    Features to drop passed as strings in list in cols_to_drop. 
    """
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # drop cols
        X = X.drop(columns=self.cols_to_drop)
        
        # convert MSSubClass to categorical
        X['MSSubClass'] = X['MSSubClass'].astype(str)

        # features with no. of bathrooms
        full_baths = ['FullBath', 'BsmtFullBath']
        half_baths = ['HalfBath', 'BsmtHalfBath']    
        # sum total bathrooms and store in TotBaths
        X['TotBaths'] = X[full_baths].sum(axis=1) + 0.5 * X[half_baths].sum(axis=1)

        # sum porch areas
        porches = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']   
        X['PorchArea'] = X[porches].sum(axis=1)
        
        # define the features with which to create the clusters
        clust_feats = ['YearBuilt', 'YearRemodAdd']
        # define k-means model
        kmeans = KMeans(n_clusters=3, n_init=50, random_state=0)
        # fit each row to a cluster and store in YrCluster column
        X['YrCluster'] = kmeans.fit_predict(X[clust_feats])

        return X
    
# transform ordinal features
class OrdinalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to label ordinal variables. 
    """
    def __init__(self):
        self=self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # label the ordinal features
        ord_vars = ['Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC'
                    , 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC']

        # don't include the column if it's been dropped
        ord_vars = [var for var in ord_vars if var in X.columns]

        ord_vars_mappings = {
            'Utilities': {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0},
            'LandSlope': {'Gtl': 2, 'Mod': 1, 'Sev': 0},
            'ExterQual': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
            'ExterCond': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
            'BsmtQual': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0},
            'BsmtCond': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0},
            'BsmtExposure': {'Gd': 4,'Av': 3,'Mn': 2,'No': 1,'None': 0},
            'BsmtFinType1': {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1,'None': 0},
            'BsmtFinType2': {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1,'None': 0},
            'HeatingQC': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
            'KitchenQual': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1},
            'Functional': {'Typ': 3, 'Min1': 2, 'Min2': 1, 'Mod': 0, 'Maj1': -1, 'Maj2': -2, 'Sev': -3, 'Sal': -4},
            'FireplaceQu': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0},
            'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0},
            'GarageQual': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0},
            'GarageCond': {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0},
            'PavedDrive': {'Y': 2, 'P': 1, 'N': 0},
            'PoolQC': {'Ex': 4,'Gd': 3,'TA': 2,'Fa': 1,'None': 0}}

        for var in ord_vars:
            X[var] = X[var].map(ord_vars_mappings[var])

        return X
    
# transformer to impute LotFrontage
class LFImputer(BaseEstimator, TransformerMixin):
    """
    Train Random Forest model to impute missing LotFrontage values. Also use SimpleImputer to impute missing values in categorical features.
    """
    def __init__(self):
        self=self

    def fit(self, X, y=None):
        simple_imputer = SimpleImputer(strategy='most_frequent')
        simple_imputer.fit(X)
        self.simple_imputer = simple_imputer
        
        return self

    def transform(self, X):
        # create a new dataframe with relevant features for predicting LotFrontage
        lf_feats = ['LotFrontage', 'LotArea', 'Neighborhood', 'BldgType', 'MSZoning', 'GarageCars']
        lf_data = X[lf_feats]
        # transform categorical variables
        lf_data = pd.get_dummies(lf_data)
        # drop rows with missing values for LotFrontage
        lf_data = lf_data.dropna(axis=0)
        lf_X = lf_data.drop(columns=['LotFrontage'])
        lf_y = lf_data.LotFrontage
        # train Random Forest model
        lf_forest_model = RandomForestRegressor(random_state=1)
        lf_forest_model.fit(lf_X, lf_y)
        # impute missing values
        lf_cols = [col for col in lf_X]
        
        def impute_lf(row):
            row.LotFrontage = lf_forest_model.predict(row[lf_cols].values.reshape(1, -1))
            return row
    
        # impute missing values for LotFrontage
        lf_X = pd.get_dummies(X)
        lf_X.loc[X.LotFrontage.isnull()] = lf_X.loc[lf_X.LotFrontage.isnull()].apply(impute_lf, axis='columns')
        X.LotFrontage = lf_X.LotFrontage
        
        # impute missing categorical features
        X[:] = self.simple_imputer.transform(X[:])
        
        return X
    
# transformer to encode cardinal features
class CardinalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to label cardinal variables using pd.get_dummies(). 
    """
    def __init__(self, cat_features):
        self.cat_features=cat_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode categorical features
        X = pd.get_dummies(X)
        # add missing features
        for col in self.cat_features:
            if col not in X:
                X[col] = 0
        #for colname in X.select_dtypes(["category", "object"]):
        #    X[colname], _ = X[colname].factorize()
        return X
      
# choose cols to drop
cols_to_drop = ['Id', 'PoolQC', 'MoSold', 'PoolQC', 'MiscFeature',
              'Alley','Fence', 'FireplaceQu', 'PoolArea', 'MiscVal',
              'LowQualFinSF','GarageYrBlt', 'GarageCond', 'GarageType',
              'GarageFinish', 'GarageQual','BsmtFinSF2', 'BsmtExposure',
              'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 
              'MasVnrArea','MasVnrType']

# create pipeline
pipeline = Pipeline(steps=[
  ('feature_selector', FeatureSelector(cols_to_drop=cols_to_drop)),
  ('neighbor_encoder', MEstimateEncoder(cols=["Neighborhood"], m=5.0)),
  ('lf_imputer', LFImputer()),
  ('ordinal_transformer', OrdinalTransformer()),
  ('cardinal_transformer', CardinalTransformer(cat_features)),
  ('standard_scaler', StandardScaler()),
])
