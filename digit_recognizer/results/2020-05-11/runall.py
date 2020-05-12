from lib_bre import *
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load train data
train_data_file = get_dataset_file_path('2020-05-04', 'train.csv')
train = pd.read_csv(train_data_file)

# Remove label
X_train = train.drop(columns='label')
y_train = train['label'].copy().to_numpy()

# Scale and reshape the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_processed = X_train_scaled.reshape((42000, 28, 28))

# Load test data
test_data_file = get_dataset_file_path('2020-05-04', 'test.csv')
X_test = pd.read_csv(test_data_file)

# Scale and reshape the test data
X_test_scaled = scaler.transform(X_test)
X_test_processed = X_test_scaled.reshape((28000, 28, 28))

# Train Model A - 200 Nodes per Layer, 2 Hidden Layers
model_a = keras.models.Sequential()
model_a.add(keras.layers.Flatten(input_shape=[28, 28]))
model_a.add(keras.layers.Dense(200, activation="relu"))
model_a.add(keras.layers.Dense(200, activation="relu"))
model_a.add(keras.layers.Dense(10, activation="softmax"))

model_a.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

history = model_a.fit(X_train_processed, y_train, epochs=2, validation_split=0.1)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("200 Nodes per Layer, 2 Hidden Layers")
plt.savefig("model_a")

# Evaluate Model A on the test set
# Make final predictions
model_a_predictions = model_a.predict_classes(X_test_processed)
model_a_predictions_series = pd.Series(model_a_predictions, name="Label")

# Output predictions
image_ids = pd.Series(data=range(1, 28001), name="ImageId")
output = pd.concat([image_ids, model_a_predictions_series], axis=1)
output.to_csv("model_a_predictions.csv", index=False)

# Train Model B - 400 Nodes per Layer, 2 Hidden Layers
model_b = keras.models.Sequential()
model_b.add(keras.layers.Flatten(input_shape=[28, 28]))
model_b.add(keras.layers.Dense(400, activation="relu"))
model_b.add(keras.layers.Dense(400, activation="relu"))
model_b.add(keras.layers.Dense(10, activation="softmax"))

model_b.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

history = model_a.fit(X_train_processed, y_train, epochs=2, validation_split=0.1)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("400 Nodes per Layer, 2 Hidden Layers")
plt.savefig("model_b")

# Train Model C - 400 Nodes per Layer, 1 Hidden Layers
model_c = keras.models.Sequential()
model_c.add(keras.layers.Flatten(input_shape=[28, 28]))
model_c.add(keras.layers.Dense(400, activation="relu"))
model_c.add(keras.layers.Dense(10, activation="softmax"))

model_c.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

history = model_a.fit(X_train_processed, y_train, epochs=2, validation_split=0.1)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("400 Nodes per Layer, 1 Hidden Layer")
plt.savefig("model_c")

# Train Model D - 200 Nodes per Layer, 1 Hidden Layers
model_d = keras.models.Sequential()
model_d.add(keras.layers.Flatten(input_shape=[28, 28]))
model_d.add(keras.layers.Dense(200, activation="relu"))
model_d.add(keras.layers.Dense(10, activation="softmax"))

model_d.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

history = model_a.fit(X_train_processed, y_train, epochs=2, validation_split=0.1)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("200 Nodes per Layer, 1 Hidden Layer")
plt.savefig("model_d")
