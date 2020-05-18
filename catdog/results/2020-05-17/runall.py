import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DATE = '2020-05-17'

# Get list of filenames
basepath = os.path.abspath('')
filepath = os.path.abspath(os.path.join(basepath, "..", "..")) + "/data/" + DATA_DATE +  "/train"
filenames = os.listdir(filepath)

# Label each image as dog or cat
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('dog')
    else:
        categories.append('cat')

# Create dataframe of filenames and labels
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# Split into train and validate datasets
train_df, validate_df = train_test_split(df, test_size=0.15, random_state=22)

# Load images
train_image_generator = ImageDataGenerator(rescale=1./255,
                                     rotation_range=30,
                                     shear_range=0.1,
                                     zoom_range=0.3,
                                     horizontal_flip=True,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1
                                     )

validate_image_generator = ImageDataGenerator(rescale=1./255)

# Define parameters for the loader
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
train_data_count = len(train_df)
validate_data_count = len(validate_df)
image_count = len(filenames)
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)


# Load training data
train_data_gen = train_image_generator.flow_from_dataframe(
    train_df,
    filepath,
    x_col='filename',
    y_col='category',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

# Load test data
validate_data_gen = validate_image_generator.flow_from_dataframe(
    validate_df,
    filepath,
    x_col='filename',
    y_col='category',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE
)


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    epochs=30,
    validation_data=validate_data_gen,
    validation_steps=validate_data_count//BATCH_SIZE,
    steps_per_epoch=image_count//BATCH_SIZE
)

# Plot learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()