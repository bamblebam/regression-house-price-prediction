# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns', None)
%matplotlib inline
# %%
dataset = pd.read_csv('dataset/train.csv')
print(dataset.shape)

# %%
dataset.head()
# %%
features_with_na = [
    features for features in dataset.columns if dataset[features].isnull().sum() >= 1]
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean()*100, 4), "%")

# %%
for feature in features_with_na:
    data = dataset.copy()
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
# %%
dataset = dataset.drop(['MiscFeature', 'PoolQC', 'Id'], axis=1)
# %%
dataset
# %%
