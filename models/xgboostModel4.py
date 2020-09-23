# xgboost model for featureEnginnering.py dataset
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
import sklearn
import pickle
pd.pandas.set_option('display.max_columns', None)

# %%
dataset = pd.read_csv('../dataset/scaled_train_3.csv')
# %%
X_train = dataset.drop(['SalePrice'], axis=1)
y_train = dataset['SalePrice']
# %%
classifier = xgboost.XGBRegressor()
classifier.fit(X_train, y_train)
# %%
filename = '../compiled_models/xgboost_2.pkl'
pickle.dump(classifier, open(filename, 'wb'))
# %%
submission_df = dataset = pd.read_csv('../dataset/sample_submission.csv')
df_test = pd.read_csv('../dataset/scaled_test_2.csv')
# %%
y_pred = classifier.predict(df_test)
# %%
pred = pd.DataFrame(y_pred)
sample = pd.concat([submission_df['Id'], pred], axis=1)
sample.columns = ['Id', 'SalePrice']
sample.to_csv('../submissions/sample_submission4.csv', index=False)
# %%
