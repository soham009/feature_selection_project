# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile,f_regression

data = pd.read_csv('data/house_prices_multivariate.csv')

# Write your solution here:

def percentile_k_features(data, k = 20):
    X = data.drop('SalePrice',axis=1)
    y = data['SalePrice']
    
    feat_col = X.columns
    fs = SelectPercentile(f_regression, percentile=k)
    
    X_new = fs.fit_transform(X, y)
    
    imp_features_kth_percentile = [feat_col[i] for i in np.argsort(fs.scores_)[::-1]]
    
    #print (imp_features_kth_percentile[:7])
    
    return imp_features_kth_percentile[:7]

percentile_k_features(data,20)


