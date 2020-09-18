# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('../dataset/dummytrain.csv')

# %%
X_train = dataset.drop(['SalePrice'], axis=1)
y_train = dataset['SalePrice']
# %%
classifier = xgboost.XGBRegressor()
classifier.fit(X_train, y_train)
# %%
