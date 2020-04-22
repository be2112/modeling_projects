import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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