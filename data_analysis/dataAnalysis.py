# data analysis and plots
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
# numerical features
numerical_features = [
    feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
dataset[numerical_features].head()
# %%
# temporal features
temporal_features = [
    feature for feature in numerical_features if "Yr" in feature or "Year" in feature]
dataset[temporal_features].head()
# %%
dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel("Year Sold")
plt.ylabel("Sale price")
plt.title("Year sold VS Sale price")
# %%
for feature in temporal_features:
    if feature != "YrSold":
        data = dataset.copy()
        data[feature] = data["YrSold"]-data[feature]
        plt.scatter(data[feature], data["SalePrice"])
        plt.xlabel(feature)
        plt.ylabel("Sale Price")
        plt.show()
# %%
# discrete features
discrete_features = [feature for feature in numerical_features if len(
    dataset[feature].unique()) < 25 and feature not in temporal_features]
print(discrete_features)
# %%
for feature in discrete_features:
    data = dataset.copy()
    data.groupby(feature)["SalePrice"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("Sale Price")
    plt.title(feature)
    plt.show()
# %%
# continuous features
continuous_features = [
    feature for feature in numerical_features if feature not in discrete_features+temporal_features]
print(continuous_features)
# %%
for feature in continuous_features:
    data = dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.title(feature)
    plt.show()
# %%
# logarithmic transformation
for feature in continuous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel("Sale Price")
        plt.title(feature)
        plt.show()
# %%
for feature in continuous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data[feature].hist(bins=25)
        plt.xlabel(feature)
        plt.ylabel("count")
        plt.title(feature)
        plt.show()

# %%
# outlier
for feature in continuous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
# %%
# categorical features
categorical_features = [
    feature for feature in dataset.columns if dataset[feature].dtypes == 'O']
print(categorical_features)
# %%
dataset[categorical_features].head()
# %%
for feature in categorical_features:
    data = dataset.copy()
    data.groupby(feature)["SalePrice"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("Sale Price")
    plt.title(feature)
    plt.show()
# %%
dataset.boxplot(column='LotFrontage')
plt.show()
# %%
