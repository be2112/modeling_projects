# runall.py
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

# Load test data
test_data_file = get_dataset_file_path('2020-05-04', 'test.csv')
X_test = pd.read_csv(test_data_file)

# Combine test and train data (Don't copy this code)
combined_dataset = pd.concat([X_train, X_test], axis=0)
image_ids = pd.Series(data=range(1, 70001), name="ImageId")
combined_dataset.index = image_ids

# Create PCA model
X_combined_scaled = StandardScaler().fit_transform(combined_dataset)
pca = PCA(n_components=0.95, whiten=True, random_state=12)
X_combined_reduced = pca.fit_transform(X_combined_scaled)

print(pca.explained_variance_.shape)

X_train_reduced = X_combined_reduced[0:42000, :]

# Build Random Forest Classifier with reduced features
rf_clf = RandomForestClassifier(oob_score=True, random_state=12)
rf_clf.fit(X_train_reduced, y_train)
score = rf_clf.oob_score_
print(rf_clf.get_params())
print(score)

# Evaluate final model on the test set
# Load test data
X_test_reduced = X_combined_reduced[42000: , :]
image_ids = pd.Series(data=range(1, 28001), name="ImageId")

# Make final predictions
rf_clf_predictions = rf_clf.predict(X_test_reduced)
rf_clf_predictions_series = pd.Series(rf_clf_predictions, name="Label")

# Output predictions
output = pd.concat([image_ids, rf_clf_predictions_series], axis=1)
output.to_csv("rf_clf_predictions.csv", index=False)
