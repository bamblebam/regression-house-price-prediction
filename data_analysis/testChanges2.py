# min max scaling and normalization for dataset from testChanges.py
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('../dataset/dummytest.csv')
# %%
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    dataset[feature] = dataset['YrSold']-dataset[feature]

# %%
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
# %%
# normalizing log
for feature in num_features:
    dataset[feature] = np.log(dataset[feature])
# %%
scaling_features = [
    feature for feature in dataset.columns]
scaler = MinMaxScaler()
scaler.fit(dataset[scaling_features])
# %%
dataset2 = pd.concat([pd.DataFrame(
    scaler.transform(dataset[scaling_features]), columns=scaling_features)], axis=1)
# %%
dataset2.drop(['Unnamed: 0'], inplace=True, axis=1)
# %%
dataset2.head()
# %%
dataset2.to_csv('../dataset/scaled_test.csv', index=False)
# %%
