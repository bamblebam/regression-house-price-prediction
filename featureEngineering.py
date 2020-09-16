# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('dataset/train.csv')
print(dataset.shape)
# %%
features_with_na = ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
features_with_na2 = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType',
                     'BsmtFinType2', 'GarageFinish', 'GarageQual', 'GarageCond']
features_with_na3 = ['LotFrontage', 'MasVnrType', 'MasVnrArea',
                     'Electrical', 'GarageYrBlt']

# %%
for feature in features_with_na2:
    dataset[feature] = dataset[feature].fillna("NO")


# %%
dataset.head()

# %%
