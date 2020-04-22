from library.library import *

import pandas as pd
import matplotlib.pyplot as plt

# Set pandas options to be more user friendly for a wide dataset
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Ignore Matplotlib memory warnings
plt.rcParams.update({'figure.max_open_warning': 100})

# Load data into a pandas dataframe.
data_file = get_dataset_file_path('2020-04-13', 'train.csv')
df = pd.read_csv(data_file)

# Get familiar with the data
df.info()
df.describe()

# Which columns have NaNs?
cols_with_nan = df.columns[df.isna().any()].tolist()
for col in cols_with_nan:
    print(col, df[col].isna().sum())

# Drop categorical variables that have over 10% of the data missing
df.drop(axis=1, labels=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)

# Fill numeric NaNs with the median value
df.fillna(df.median(), inplace=True)

# Drop remaining rows with NaNs
df.dropna(inplace=True)

# Produce LOTS of histograms
df_columns = list(df.columns.values)
for column in df_columns:
    fig, ax = plt.subplots()
    ax.hist(df[column])
    ax.set_ylabel('Count', fontdict={'fontsize': 20})
    ax.set_xlabel(column, fontdict={'fontsize': 20})
    ax.set_title(column + ' Histogram', fontdict={'fontsize': 20})
    fig.savefig(column + '_Histogram', format='png')

# Check out correlations
corr_matrix = df.corr()
