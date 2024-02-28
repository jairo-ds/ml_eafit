import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import datasets, cluster

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import wandb
wandb.login()

# Load data
housing = datasets.fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X, y = X[::2], y[::2]  # subsample for faster demo
wandb.errors.term._show_warnings = False
# ignore warnings about charts being built from subset of data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model, get predictions
reg = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

run = wandb.init(project='firstmodel', name="regression")

wandb.sklearn.plot_residuals(reg, X_train, y_train)

wandb.sklearn.plot_outlier_candidates(reg, X_train, y_train)

wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')

wandb.finish()
#j