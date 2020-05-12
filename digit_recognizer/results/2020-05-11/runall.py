from lib_bre import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

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


def _log(message):
    print('[SimpleTimeTracker] {function_name} {total_time:.3f}'.format(**message))


@simple_time_tracker(_log)
def train_model(hidden_layers, nodes_per_layer, model_name):
    nn_clf = keras.models.Sequential()
    nn_clf.add(keras.layers.Flatten(input_shape=[28, 28]))

    for i in range(hidden_layers):
        nn_clf.add(keras.layers.Dense(nodes_per_layer, activation="relu"))

    nn_clf.add(keras.layers.Dense(10, activation="softmax"))
    nn_clf.compile(loss="sparse_categorical_crossentropy",
                   optimizer="sgd",
                   metrics=["accuracy"])

    history = nn_clf.fit(X_train_processed, y_train, epochs=30, validation_split=0.1)

    # Plot learning curves
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title(str(nodes_per_layer) + " Nodes per Layer, " + str(hidden_layers) + " Hidden Layers")
    plt.savefig(model_name)

    # Make final predictions on test set
    model_predictions = nn_clf.predict_classes(X_test_processed)
    model_predictions_series = pd.Series(model_predictions, name="Label")

    # Output predictions
    image_ids = pd.Series(data=range(1, 28001), name="ImageId")
    output = pd.concat([image_ids, model_predictions_series], axis=1)
    output.to_csv(model_name + "_predictions.csv", index=False)


train_model(2, 200, "model_a")
train_model(2, 400, "model_b")
train_model(1, 400, "model_c")
train_model(1, 200, "model_d")
