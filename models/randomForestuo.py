# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
import sklearn
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from hyperopt.pyll.base import scope
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK
pd.pandas.set_option('display.max_columns', None)

# %%
dataset = pd.read_csv('../dataset/dummytrain.csv')
# %%
X_train = dataset.drop(['SalePrice'], axis=1)
y_train = dataset['SalePrice']
submission_df = dataset = pd.read_csv('../dataset/sample_submission.csv')
df_test = pd.read_csv('../dataset/dummytest.csv')
# %%
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(df_test)
print(y_pred)

# %%
filename = '../compiled_models/random_forest2.pkl'
pickle.dump(regressor, open(filename, 'wb'))
# %%
pred = pd.DataFrame(y_pred)
sample = pd.concat([submission_df['Id'], pred], axis=1)
sample.columns = ['Id', 'SalePrice']
sample.to_csv('../submissions/sample_submission7.csv', index=False)
# %%
