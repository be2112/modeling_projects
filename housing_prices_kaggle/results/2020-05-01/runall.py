# runall.py

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder

from lib_bre import *
from lib_bre import ColumnSelector, TypeSelector, CategoricalEncoder

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
        )),
        ("categorical_features", make_pipeline(
            TypeSelector("category"),
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder()
        ))
    ])
)

X_train = preprocess_pipeline.fit_transform(X_train)

gb_param_grid = [
    {"max_features": ["auto", "log2", "sqrt"],
     "max_depth": [1, 3, 10, 20, 40, 80],
     "learning_rate": [0.001, 0.01, 0.1],
     "n_estimators": [100, 200, 400, 800, 1600, 3200, 6400],
     "random_state": [79]}
]
rf_param_grid = [
    {"max_features": ["auto", "log2", "sqrt"],
     "max_depth": [1, 3, 10, 20, 40, 80],
     "n_estimators": [100, 200, 400, 800, 1600, 3200, 6400],
     "random_state": [79]}
]

gb_grid_search = GridSearchCV(GradientBoostingRegressor(), gb_param_grid, cv=10, scoring="neg_mean_squared_error",
                              n_jobs=-1,
                              return_train_score=True)
gb_grid_search.fit(X_train, y_train)
print(gb_grid_search.best_estimator_)

gb_final_model = gb_grid_search.best_estimator_

rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=10, scoring="neg_mean_squared_error",
                              n_jobs=-1,
                              return_train_score=True)
rf_grid_search.fit(X_train, y_train)
print(rf_grid_search.best_estimator_)

rf_final_model = rf_grid_search.best_estimator_

# Evaluate final model on training data
gb_scores = cross_val_score(gb_final_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
gb_reg_rmse_scores = np.sqrt(-gb_scores)
print("GB Scores")
display_scores(gb_reg_rmse_scores)

rf_scores = cross_val_score(rf_final_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
rf_reg_rmse_scores = np.sqrt(-rf_scores)
print("RF Scores")
display_scores(rf_reg_rmse_scores)
# Evaluate final model on the test set

# Load test data
data_file = get_dataset_file_path('2020-04-13', 'test.csv')
X_test = pd.read_csv(data_file)
output = X_test["Id"]

# Preprocess data
X_test = preprocess_pipeline.transform(X_test)

# Make final predictions
gb_final_predictions = gb_final_model.predict(X_test)
rf_final_predictions = rf_final_model.predict(X_test)

# Output predictions
gb_final_predictions = pd.Series(gb_final_predictions, name="SalePrice")
rf_final_predictions = pd.Series(rf_final_predictions, name="SalePrice")

output = pd.concat([output, gb_final_predictions], axis=1)
output.to_csv("gb_predictions.csv", index=False)
output = pd.concat([output, rf_final_predictions], axis=1)
output.to_csv("rf_predictions.csv", index=False)
