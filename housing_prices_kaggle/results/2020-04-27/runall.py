# runall.py
from collections import OrderedDict

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from lib_bre import *
from lib_bre import ColumnSelector, TypeSelector, CategoricalEncoder
import matplotlib.pyplot as plt

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

stump_forest = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('decision_tree', RandomForestRegressor(warm_start=True, max_depth=1, oob_score=True, random_state=79, n_jobs=-1))
])

five_deep_forest = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('decision_tree', RandomForestRegressor(warm_start=True, max_depth=5, oob_score=True, random_state=79, n_jobs=-1))
])

ten_deep_forest = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('decision_tree', RandomForestRegressor(warm_start=True, max_depth=10, oob_score=True, random_state=79, n_jobs=-1))
])

twenty_deep_forest = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('decision_tree', RandomForestRegressor(warm_start=True, max_depth=20, oob_score=True, random_state=79, n_jobs=-1))
])

forty_deep_forest = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('decision_tree', RandomForestRegressor(warm_start=True, max_depth=40, oob_score=True, random_state=79, n_jobs=-1))
])

eighty_deep_forest = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('decision_tree', RandomForestRegressor(warm_start=True, max_depth=80, oob_score=True, random_state=79, n_jobs=-1))
])

ensemble_regressors = [
    ("RandomForestRegressor, max_depth=1", stump_forest),
    ("RandomForestRegressor, max_depth=5", five_deep_forest),
    ("RandomForestRegressor, max_depth=10", ten_deep_forest),
    ("RandomForestRegressor, max_depth=20", twenty_deep_forest),
    ("RandomForestRegressor, max_depth=40", forty_deep_forest),
    ("RandomForestRegressor, max_depth=80", eighty_deep_forest)
]

error_rate = OrderedDict((label, []) for label, _ in ensemble_regressors)

# range of 'n_trees' to explore'
min_trees = 20
max_trees = 200

for label, reg in ensemble_regressors:
    for i in range(min_trees, max_trees + 1):
        reg.steps[1][1].set_params(n_estimators=i)
        reg.fit(X_train, y_train)

        oob_error = 1 - reg.steps[1][1].oob_score_
        error_rate[label].append((i, oob_error))

# Generate test error vs n_trees plot
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_trees, max_trees)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.savefig('test_errors.png', dpi=100)
plt.show()
