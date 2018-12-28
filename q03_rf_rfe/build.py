# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Your solution code here

def rf_rfe(data):
    X = data.drop('SalePrice',axis=1)
    y = data['SalePrice']
    random_forest_model = RandomForestClassifier()
    
    rfe = RFE(random_forest_model,n_features_to_select=len(X.columns)/2)
    rfe = rfe.fit(X,y)
    
    #print (list(X.columns[rfe.support_]))
    
    return list(X.columns[rfe.support_])

rf_rfe(data)



