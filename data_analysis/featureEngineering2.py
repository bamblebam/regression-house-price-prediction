# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('../dataset/dummytrain.csv')
# %%
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    dataset[feature] = dataset['YrSold']-dataset[feature]

# %%
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
# %%
# normalizing log
for feature in num_features:
    dataset[feature] = np.log(dataset[feature])
# %%
