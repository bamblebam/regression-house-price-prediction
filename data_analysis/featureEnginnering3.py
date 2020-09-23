# min max scaling and normalization for dataset from featureEnginnering.py without sp
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('../dataset/dummytrain.csv')
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
    feature for feature in dataset.columns if feature not in ['SalePrice']]
scaler = MinMaxScaler()
scaler.fit(dataset[scaling_features])
# %%
dataset2 = pd.concat([dataset['SalePrice'].reset_index(drop=True), pd.DataFrame(
    scaler.transform(dataset[scaling_features]), columns=scaling_features)], axis=1)
# %%
dataset2.drop(['Unnamed: 0'], inplace=True, axis=1)
# %%
dataset2.head()
# %%
dataset2.to_csv('../dataset/scaled_train_4.csv', index=False)
# %%
