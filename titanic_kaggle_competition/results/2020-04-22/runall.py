import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder

from library.library import *
from library.transformers import ColumnSelector, TypeSelector, CategoricalEncoder

# Load training data

train_data_filepath = get_dataset_file_path('2020-04-04', 'train.csv')
train_df = pd.read_csv(train_data_filepath)

# Remove label
X_train = train_df.drop(columns='Survived')
y_train = train_df['Survived'].copy()

# Variables to use in the model
x_cols = ['Pclass',
          'Sex',
          'Age',
          'SibSp',
          'Parch',
          'Fare',
          'Embarked']

# Build pre-processing pipeline
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
            OrdinalEncoder()
        ))
    ])
)

# Preprocess data
X_train = preprocess_pipeline.fit_transform(X_train)

# Instantiate and train a simple Naive Bayes classifier model
nb = GaussianNB()

# Generate cross-validated estimates for each data point
y_train_pred_proba = cross_val_predict(nb, X_train, y_train, cv=10, method="predict_proba")

# Compute precision and recall for all possible thresholds
precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_pred_proba[:, 1])

# Plot precision vs recall graph
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.savefig('Precision_Recall_plot.svg')
plt.show()

# Generate precision and recall stats
y_train_pred_bool = y_train_pred_proba[:, 1] > 0.5
y_train_pred_binary = y_train_pred_bool.astype(int)
precision_score = precision_score(y_train, y_train_pred_binary)
recall_score = recall_score(y_train, y_train_pred_binary)
print("Precision Score:", precision_score)
print("Recall Score:", recall_score)

# Generate accuracy stats
accuracy = sum(y_train_pred_binary == y_train) / y_train.size
print("Accuracy: ", accuracy)

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_proba[:, 1])
plot_roc_curve(fpr, tpr)
plt.savefig('AUC_plot.svg')
plt.show()

# Calculate area under the curve
area_under_curve = auc(fpr, tpr)
print("Area under the curve: ", area_under_curve)

# Generate final predictions on test data

# Load test data
data_file = get_dataset_file_path('2020-04-04', 'test.csv')
X_test = pd.read_csv(data_file)
output = X_test["PassengerId"]

# Preprocess data
X_test = preprocess_pipeline.transform(X_test)

# Fit model
final_model = GaussianNB().fit(X_train, y_train)

# Make final predictions
final_predictions = final_model.predict(X_test)

# Output predictions
final_predictions = pd.Series(final_predictions, name="Survived")
output = pd.concat([output, final_predictions], axis=1)
output.to_csv("predictions.csv", index=False)
