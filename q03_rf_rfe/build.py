# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
def rf_rfe(df):
    y= df['SalePrice']
    X = df.loc[:,df.columns != 'SalePrice']
    model = RandomForestClassifier()


    rfe = RFE(estimator = model,n_features_to_select = len(df.columns)/2)
    rfe = rfe.fit(X, y)
    support = np.array(rfe.support_)

    features = np.array(X.columns.values)
    a = features[support].tolist()
    return a





rf_rfe(data)
