# basic feature engineering
# changed na values and made dummies

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
# replacing categorical features with dummy variables
categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# %%
main_dataset = dataset.copy()
test_data = pd.read_csv('dataset/updatedtest.csv')
# %%
final_data = pd.concat([main_dataset, test_data])
print(final_data.shape)
# %%
dataset2 = final_data.copy()
# %%
for feature in categorical_features:
    temp_df = pd.get_dummies(
        final_data[feature], prefix=feature, drop_first=True)
    dataset2.drop([feature], axis=1, inplace=True)
    dataset2 = pd.concat([dataset2, temp_df], axis=1)

# %%
print(dataset.shape)
# %%
df_train = dataset2.iloc[:1460]
df_test = dataset2.iloc[1460:]

# %%
df_test.drop(['Id', 'SalePrice'], axis=1, inplace=True)
df_train.drop(['Id'], axis=1, inplace=True)
# %%
df_test.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)
# %%
df_test.head()
# %%
df_test.to_csv('dataset/dummytest.csv')
df_train.to_csv('dataset/dummytrain.csv')
# %%
