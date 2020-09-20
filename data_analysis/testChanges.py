# same procedure as featureEngineering.py but for train data
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
dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
dataset['GarageArea'] = dataset['GarageArea'].fillna(0)
dataset['KitchenQual'] = dataset['KitchenQual'].fillna(
    dataset['KitchenQual'].mode()[0])
dataset['Functional'] = dataset['Functional'].fillna(
    dataset['Functional'].mode()[0])
dataset['SaleType'] = dataset['SaleType'].fillna(
    dataset['SaleType'].mode()[0])
# %%
# replacing categorical features with dummy variables
categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
# %%
# %%
dataset2 = dataset.copy()

# %%
dataset2.to_csv('dataset/updatedtest.csv', index=False)
