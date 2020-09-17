# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('dataset/train.csv')
# %%
# features with na value
features_with_na = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
# features with na value in description
features_with_na2 = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType',
                     'BsmtFinType2', 'GarageFinish', 'GarageQual', 'GarageCond']
# features with na value due to no data
features_with_na3 = ['LotFrontage', 'MasVnrType', 'MasVnrArea',
                     'Electrical', 'GarageYrBlt']

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
