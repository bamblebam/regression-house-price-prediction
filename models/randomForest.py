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
from sklearn.metrics import mean_squared_log_error
from hyperopt.pyll.base import scope
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
pd.pandas.set_option('display.max_columns', None)

# %%
dataset = pd.read_csv('../dataset/dummytrain.csv')
# %%
X_train = dataset.drop(['SalePrice'], axis=1)
y_train = dataset['SalePrice']
# %%
parameter_space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700]),
    'max_depth': hp.randint('max_depth', 70, 100),
    'max_features': hp.randint('max_features', 2, 5),
    'criterion': hp.choice('criterion', ['mse', 'mae']),
}
# %%
submission_df = dataset = pd.read_csv('../dataset/sample_submission.csv')
df_test = pd.read_csv('../dataset/dummytest.csv')

# %%
train_X, test_X, train_y, test_y = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)
# %%


def HPTuning(params):
    classifier = RandomForestRegressor(**params, n_jobs=1)
    classifier.fit(train_X, train_y)
    y_pred = classifier.predict(test_X)
    score = np.sqrt(mean_squared_log_error(test_y, y_pred))
    return {"loss": score, 'status': STATUS_OK}


# %%
trials = Trials()
best = fmin(
    fn=HPTuning,
    space=parameter_space,
    algo=tpe.suggest,
    max_evals=25,
    trials=trials
)
print("Best: {}".format(best))

# %%
parameter_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# %%
estimator = RandomForestRegressor()
grid_search = GridSearchCV(
    estimator=estimator, param_grid=parameter_grid, n_jobs=-1, cv=3, verbose=2)
# %%
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
# %%
regressor = RandomForestRegressor(
    max_depth=110, max_features=3, n_estimators=200, criterion='mse', min_samples_leaf=3, min_samples_split=8)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(df_test)
print(y_pred)

# %%
filename = '../compiled_models/random_forest4.pkl'
pickle.dump(regressor, open(filename, 'wb'))
# %%
pred = pd.DataFrame(y_pred)
sample = pd.concat([submission_df['Id'], pred], axis=1)
sample.columns = ['Id', 'SalePrice']
sample.to_csv('../submissions/sample_submission9.csv', index=False)
# %%
