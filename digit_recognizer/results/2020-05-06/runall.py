# 2020-05-06 runall.py
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from lib_bre import *


# Load train data
train_data_file = get_dataset_file_path('2020-05-04', 'train.csv')
train = pd.read_csv(train_data_file)

# Remove label
X_train = train.drop(columns='label')
y_train = train['label'].copy()

# Create PCA model
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
pca = PCA(n_components=0.95, whiten=True, random_state=12)
X_train_reduced = pca.fit_transform(X_train_scaled)

print(pca.explained_variance_.shape)

# Build Random Forest Classifier with reduced features
rf_clf = RandomForestClassifier(oob_score=True, random_state=12)
rf_clf.fit(X_train_reduced, y_train)
score = rf_clf.oob_score_
print(rf_clf.get_params())
print(score)

# Evaluate final model on the test set
# Load test data
test_data_file = get_dataset_file_path('2020-05-04', 'test.csv')
X_test = pd.read_csv(test_data_file)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Reduce test data with PCA
X_test_reduced = pca.transform(X_test_scaled)

# Make final predictions
rf_clf_predictions = rf_clf.predict(X_test_reduced)
rf_clf_predictions_series = pd.Series(rf_clf_predictions, name="Label")

# Output predictions
image_ids = pd.Series(data=range(1, 28001), name="ImageId")
output = pd.concat([image_ids, rf_clf_predictions_series], axis=1)
output.to_csv("rf_clf_predictions.csv", index=False)
