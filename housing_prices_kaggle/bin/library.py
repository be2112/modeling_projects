# libary.py
import os
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def get_dataset_file_path(date: object, filename: object) -> object:
    """Produces a filepath for the dataset.

    :parameter date (string): The date folder name.  Ex: "2020-02-05"
    :parameter filename (string): The csv filename.
    :returns filepath (string): The filepath for the dataset.

    Example:

    project_root
    ├── README.md
    ├── data
    │   └── 2020-04-13
    │       ├── README.md
    │       ├── data_description.txt
    │       ├── test.csv
    │       └── train.csv
    ├── docs
    ├── requirements.yml
    └── results
        └── 2020-04-13
            └── runall.py

    The function is called from the 'runall.py' file.
    >> get_data_file_path('2020-04-13', 'train.csv')
    '~/project_root/data/2020-04-13/train.csv'
    """

    basepath = os.path.abspath('')
    filepath = os.path.abspath(os.path.join(basepath, "..", "..")) + "/data/" + date + "/" + filename
    return filepath

def convert_object_to_categorical(df):
    """Converts columns in a pandas dataframe of dtype 'object' to dtype 'categorical.'  This is a destructive method

    :parameter df (pandas dataframe): A pandas dataframe
    """
    assert isinstance(df, pd.DataFrame)

    object_columns = df.select_dtypes(include='object').columns.tolist()
    for obj_col in object_columns:
        df[obj_col] = df[obj_col].astype('category')

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TypeSelector(BaseEstimator, TransformerMixin):
    """Returns a dataframe that only includes columns of the specified datatype.

    :parameter dtype (string):  The datatype to filter on.
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=self.dtype)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Converts columns in a pandas dataframe of dtype 'object' to dtype 'categorical.'
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        object_columns = X.select_dtypes(include='object').columns.tolist()
        for obj_col in object_columns:
            X[obj_col] = X[obj_col].astype('category')
        return X


class DropNaNs(BaseEstimator, TransformerMixin):
    """Drops all NaNs in a pandas dataframe.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        return X.dropna(inplace=True)
