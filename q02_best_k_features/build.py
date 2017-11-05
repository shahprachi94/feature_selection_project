# %load q02_best_k_features/build.py
# Default imports
import numpy as np
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
def percentile_k_features(df, k= 20):

    X = df.loc[:, df.columns != 'SalePrice']
    y = df['SalePrice']
    features = X.columns.values.tolist()
    fs = SelectPercentile(f_regression, percentile=20)
    X_train_fs = fs.fit_transform(X,y)

    new = []

    sorted_scores = np.argsort(fs.scores_)[::-1]
    b = np.sum(fs.get_support())

    for i in sorted_scores:
        new.append(features[i])

    return new[:b]
