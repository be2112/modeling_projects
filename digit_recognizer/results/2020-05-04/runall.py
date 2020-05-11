# 2020-05-04 runall.py
from sklearn.ensemble import RandomForestClassifier
from lib_bre import *


# Load training data
data_file = get_dataset_file_path('2020-05-04', 'train.csv')
train = pd.read_csv(data_file)

# Remove label
X_train = train.drop(columns='label')
y_train = train['label'].copy()

rf_clf = RandomForestClassifier(random_state=98, oob_score=True)
rf_clf.fit(X_train, y_train)
score = rf_clf.oob_score_
print(rf_clf.get_params())
print(score)

# Evaluate final model on the test set
# Load test data
test_data_file = get_dataset_file_path('2020-05-04', 'test.csv')
X_test = pd.read_csv(test_data_file)
image_ids = pd.Series(data=range(1, 28001), name="ImageId")

# Make final predictions
rf_clf_predictions = rf_clf.predict(X_test)
rf_clf_predictions_series = pd.Series(rf_clf_predictions, name="Label")

# Output predictions
output = pd.concat([image_ids, rf_clf_predictions_series], axis=1)
output.to_csv("rf_clf_predictions.csv", index=False)
