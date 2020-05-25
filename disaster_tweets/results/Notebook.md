# Cat Dog Lab Notebook

## 2020-05-17

* Define the objective in business terms.
  * We are providing advice to a website provider that wants to automatically classify images.  The goal is the highest possible accuracy.  We are OK with sacrificing training time to get better accuracy.
* How will your solution be used?
  * The solution will be used to classify images that end users submit.
* What are the current solutions/workarounds (if any)?
  * N/A
* How should you frame this problem (supervised/unsupervised, online/offline, etc.)?
  * This is a supervised learning problem.
* How should performance be measured?
  * Accuracy
* Is the performance measure aligned with the business objective?
  * Yes

### Experiment Plan

My plan is to build a pretty standard CNN, and get it to work.

### Modeling Decisions

```python
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
```



### Experiment Results

loss: 0.3439

accuracy: 0.8508

val_loss: 0.3559

val_accuracy: 0.8483

### Next Steps

This model took over a full day to train.  I don't have time to train four of these models.  So I need to change my plan.  Next I'll try to start with a pre-trained CNN.  Use that to generate features.  Then feed those into a small neural network.



## 2020-05-18

### Experiment Plan

* Import an Xception model that has been pretrained on ImageNet.
* Use that model to extract features from the labeled pictures.
* Then feed those features into a small neural network.
* First layer: ```Dense(256, activation='relu', input_dim=7 * 7 * 2048)```
* Second layer: ```Dropout(0.5)```
* Third layer: ```(Dense(1, activation='sigmoid')```
* For the 2x2 factorial design:
  * 2 models will have the dropout layer.  2 models will not have the dropout layer.
  * 2 models will be trained with 30 epochs.  While 2 models will be trained with 10 epochs.

### Experiment Results and Interpretation

| Dropout Layer? | # of Epochs | Validation Loss | Training Set Accuracy | Validation Accuracy | Kaggle Score (Log Loss) |
| -------------- | ----------- | --------------- | --------------------- | ------------------- | ----------------------- |
| Y              | 30          | 0.3054          | 0.9993                | 0.9883              | 0.40068                 |
| Y              | 10          | 0.1760          | 0.9954                | 0.9867              | 0.35403                 |
| N              | 30          | 0.3605          | 0.9999                | 0.966               | 0.42839                 |
| N              | 10          | 0.2771          | 0.9979                | 0.9856              | 0.42839                 |

Clearly adding in the dropout layer had a significant positive impact.  30 epochs vs 10 epochs had mixed results.



### Next Steps

I could further improve this model by implementing data augmentation to increase the size of my dataset, and to increase the robustness of my model.