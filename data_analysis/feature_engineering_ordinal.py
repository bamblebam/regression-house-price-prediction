# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.pandas.set_option('display.max_columns', None)
# %%
dataset = pd.read_csv('../dataset/na_train.csv')
