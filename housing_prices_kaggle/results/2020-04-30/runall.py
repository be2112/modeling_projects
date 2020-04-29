# runall.py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder

from library.library import *
from library.transformers import ColumnSelector, TypeSelector, CategoricalEncoder

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

gb_1 = GradientBoostingRegressor(warm_start=True, max_features='auto', max_depth=1, learning_rate=0.1, random_state=79,
                                 n_estimators=1000)
gb_2 = GradientBoostingRegressor(warm_start=True, max_features='auto', max_depth=3, learning_rate=0.1, random_state=79,
                                 n_estimators=1000)
gb_3 = GradientBoostingRegressor(warm_start=True, max_features='auto', max_depth=10, learning_rate=0.1, random_state=79,
                                 n_estimators=1000)
gb_4 = GradientBoostingRegressor(warm_start=True, max_features='auto', max_depth=1, learning_rate=0.01, random_state=79,
                                 n_estimators=1000)
gb_5 = GradientBoostingRegressor(warm_start=True, max_features='auto', max_depth=3, learning_rate=0.01, random_state=79,
                                 n_estimators=1000)
gb_6 = GradientBoostingRegressor(warm_start=True, max_features='auto', max_depth=10, learning_rate=0.01,
                                 random_state=79, n_estimators=1000)
gb_7 = GradientBoostingRegressor(warm_start=True, max_features='log2', max_depth=1, learning_rate=0.1, random_state=79,
                                 n_estimators=1000)
gb_8 = GradientBoostingRegressor(warm_start=True, max_features='log2', max_depth=3, learning_rate=0.1, random_state=79,
                                 n_estimators=1000)
gb_9 = GradientBoostingRegressor(warm_start=True, max_features='log2', max_depth=10, learning_rate=0.1, random_state=79,
                                 n_estimators=1000)
gb_10 = GradientBoostingRegressor(warm_start=True, max_features='log2', max_depth=1, learning_rate=0.01,
                                  random_state=79, n_estimators=1000)
gb_11 = GradientBoostingRegressor(warm_start=True, max_features='log2', max_depth=3, learning_rate=0.01,
                                  random_state=79, n_estimators=1000)
gb_12 = GradientBoostingRegressor(warm_start=True, max_features='log2', max_depth=10, learning_rate=0.01,
                                  random_state=79, n_estimators=1000)

ensemble_regressors = [
    ("GB, max_features=auto, max_depth=1, learning_rate=0.1", gb_1),
    ("GB, max_features=auto, max_depth=3, learning_rate=0.1", gb_2),
    ("GB, max_features=auto, max_depth=10, learning_rate=0.1", gb_3),
    ("GB, max_features=auto, max_depth=1, learning_rate=0.01", gb_4),
    ("GB, max_features=auto, max_depth=3, learning_rate=0.01", gb_5),
    ("GB, max_features=auto, max_depth=10, learning_rate=0.01", gb_6),
    ("GB, max_features=log2, max_depth=1, learning_rate=0.1", gb_7),
    ("GB, max_features=log2, max_depth=3, learning_rate=0.1", gb_8),
    ("GB, max_features=log2, max_depth=10, learning_rate=0.1", gb_9),
    ("GB, max_features=log2, max_depth=1, learning_rate=0.01", gb_10),
    ("GB, max_features=log2, max_depth=3, learning_rate=0.01", gb_11),
    ("GB, max_features=log2, max_depth=10, learning_rate=0.01", gb_12)
]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=8)

ensemble_gbrt_best = []
ensemble_min_error = []

for label, reg in ensemble_regressors:
    reg.fit(X_train, y_train)
    errors = [mean_squared_error(y_val, y_pred) for y_pred in reg.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors) + 1
    gbrt_best = reg.set_params(n_estimators=bst_n_estimators)
    ensemble_gbrt_best.append(gbrt_best)
    min_error = np.min(errors)
    ensemble_min_error.append(min_error)

    plt.plot(errors, "b.-")
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
    plt.plot([0, 1000], [min_error, min_error], "k--")
    plt.plot(bst_n_estimators, min_error, "ko")
    plt.text(bst_n_estimators, min_error * 1.2, "Minimum", ha="center", fontsize=14)
    plt.xlabel("Number of trees")
    plt.ylabel("Error", fontsize=16)
    plt.title(label + "Validation error", fontsize=8)

    plt.savefig(label + 'test_errors.png', dpi=400)
    plt.show()

# Find best model
best_index = np.argmin(ensemble_min_error)
overall_best_model = ensemble_gbrt_best[best_index]
print(overall_best_model)
