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
# %%
parameter_space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600, 700]),
    'max_depth': hp.randint('max_depth', 70, 100),
    'max_features': hp.randint('max_features', 2, 5),
    'criterion': hp.choice('criteraion', ['mse', 'mae']),
}

# %%


def HPTuning(params):
    classifier = RandomForestRegressor(**params, n_jobs=1)
    accuracy = cross_val_score(
        classifier, X_train, y_train, scoring="neg_mean_absolute_error").mean()
    return {"loss": -accuracy, 'status': STATUS_OK}


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
