# runall.py
from bin.library import *
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV

import pandas as pd
import numpy as np

# Load training data
data_file = get_dataset_file_path('2020-04-13', 'train.csv')
train = pd.read_csv(data_file)

# Remove label
X_train = train.drop(columns='SalePrice')
y_train = train['SalePrice'].copy()

# Variables to use in the model
x_cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities',
          'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
          'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
          'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
          'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
          'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
          'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
          'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
          'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
          'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

# Build preprocessing pipeline
preprocess_pipeline = make_pipeline(
    ColumnSelector(columns=x_cols),
    CategoricalEncoder(),
    FeatureUnion(transformer_list=[
        ("numeric_features", make_pipeline(
            TypeSelector(np.number),
            SimpleImputer(missing_values=np.nan, strategy='median'),
            RobustScaler()
        )),
        ("categorical_features", make_pipeline(
            TypeSelector("category"),
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder()
        ))
    ])
)

# Preprocess data
X_train = preprocess_pipeline.fit_transform(X_train)

# Instantiate a simple Ridge Regression model and assess its accuracy using a 10-fold cross validation
ridge_reg = Ridge()
scores = cross_val_score(ridge_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
ridge_reg_rmse_scores = np.sqrt(-scores)
display_scores(ridge_reg_rmse_scores)

# Instantiate a simple Lasso Regression model and assess its accuracy using a 10-fold cross validation
lasso_reg = Lasso()
scores = cross_val_score(lasso_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
lasso_reg_rmse_scores = np.sqrt(-scores)
display_scores(lasso_reg_rmse_scores)

# Instantiate a simple Elastic Net Regression model and assess its accuracy using a 10-fold cross validation
elastic_net = ElasticNet()
scores = cross_val_score(elastic_net, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
elastic_net_reg_rmse_scores = np.sqrt(-scores)
display_scores(elastic_net_reg_rmse_scores)

# Fine tune the Ridge Regression model using a randomized search cross validation
param_grid = [
    {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
     "fit_intercept": [False],
     "normalize": [True, False],
     "max_iter": [10, 100, 1000, 1000],
     "tol": [0.01, 0.001, 0.001],
     "solver": ["cholesky"]}
]

grid_search = GridSearchCV(Ridge(), param_grid, cv=10, scoring="neg_mean_squared_error", n_jobs=4,
                           return_train_score=True)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)

final_model = grid_search.best_estimator_

# Evaluate final model on training data
scores = cross_val_score(final_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
ridge_reg_rmse_scores = np.sqrt(-scores)
display_scores(ridge_reg_rmse_scores)

# Evaluate final model on the test set

# Load test data
data_file = get_dataset_file_path('2020-04-13', 'test.csv')
X_test = pd.read_csv(data_file)
output = X_test["Id"]

# Preprocess data
X_test = preprocess_pipeline.transform(X_test)

# Make final predictions
final_predictions = final_model.predict(X_test)

# Output predictions
final_predictions = pd.Series(final_predictions, name="SalePrice")
output = pd.concat([output, final_predictions], axis=1)
output.to_csv("predictions.csv", index=False)
