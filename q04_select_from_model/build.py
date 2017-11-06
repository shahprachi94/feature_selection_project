# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')
def select_from_model(df):
    y= df['SalePrice']
    X = df.loc[:,df.columns != 'SalePrice']
    np.random.seed = 9
    model = RandomForestClassifier()
    sfm = SelectFromModel(estimator = model)
    sfm = sfm.fit(X,y)
    support = sfm.get_support()
    features = np.array(X.columns.values)
    a = features[support].tolist()
    return a
