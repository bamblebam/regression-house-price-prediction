# used ordinal encoder
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('../dataset/na_train.csv')
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
categorical_features = [
    feature for feature in dataset.columns if dataset[feature].dtypes == "O"]
# %%
encoder = OrdinalEncoder()
dataset[categorical_features] = encoder.fit_transform(
    dataset[categorical_features])
# %%
scaling_features = [
    feature for feature in dataset.columns if feature not in ['SalePrice']]
scaler = MinMaxScaler()
scaler.fit(dataset[scaling_features])
# %%
dataset2 = pd.concat([dataset['SalePrice'].reset_index(drop=True), pd.DataFrame(
    scaler.transform(dataset[scaling_features]), columns=scaling_features)], axis=1)

# %%
dataset2.drop(['Id'], axis=1, inplace=True)
# %%
dataset2.to_csv('../dataset/scaled_train_2.csv', index=False)
# %%
