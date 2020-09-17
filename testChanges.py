# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('dataset/test.csv')

# %%

features_with_na = ['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                    'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']
features_with_na2 = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType',
                     'BsmtFinType2', 'GarageFinish', 'GarageQual', 'GarageCond']
features_with_na3 = ['MSZoning', 'LotFrontage', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                     'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'SaleType']

# %%
for feature in features_with_na2:
    dataset[feature] = dataset[feature].fillna("None")

# %%
dataset['LotFrontage'] = dataset['LotFrontage'].fillna(
    np.round(dataset['LotFrontage'].mean()))
dataset['MasVnrType'] = dataset['MasVnrType'].fillna("None")
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
dataset['Electrical'] = dataset['Electrical'].fillna(
    dataset['Electrical'].mode()[0])
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['YearBuilt'])

# %%
dataset['MSZoning'] = dataset['MSZoning'].fillna(
    dataset['MSZoning'].mode()[0])
dataset['Utilities'] = dataset['Utilities'].fillna(
    dataset['Utilities'].mode()[0])
dataset['Exterior1st'] = dataset['Exterior1st'].fillna(
    dataset['Exterior1st'].mode()[0])
dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior1st'])
dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(0)
dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(0)
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(0)
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(0)

# %%
dataset.loc[dataset['TotalBsmtSF'] == 0, 'BsmtFullBath'] = 0
dataset.loc[dataset['TotalBsmtSF'] == 0, 'BsmtHalfBath'] = 0
# %%
features_with_na = [
    features for features in dataset.columns if dataset[features].isnull().sum() >= 1]
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean()*100, 4), "%")
# %%
