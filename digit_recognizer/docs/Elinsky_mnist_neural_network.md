Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
12 May 2020


# Assignment 6: Neural Networks

## Problem Description
Given an array of pixels and their associated degree of darkness, the objective is to classify each image as the correct digit.  As an employee at a financial institution, I need to assess how accurate neural networks are at performing this task, along with recommending an architecture.
## Research Design and Modeling Methods
For this experiment, my plan was to fit the MNIST dataset with a neural network model.  I would conduct a 2x2 factorial design, comparing 1 vs 2 hidden layers and 200 vs 400 neurons per layer.  I didn't plan on conducting any automated hyperparameter tuning via a grid search. 
## Data Preparation and Model Architecture
I used a MinMaxScaler to scale the features to the [0,1] range.  This is necessary because I am training the neural network using Gradient Descent.  I reshaped the array from (42000, 784) to (42000, 28, 28).  Each of the four models are Keras sequential models.  This is a fairly simple model.  Each layer has one input tensor and one output tensor.  The first layer is a Flatten layer.  The second and third layers are Dense layers with relu activation.  The final layer is a Dense layer with only 10 nodes, and softmax activation.  Softmax activation is necessary because the outputs should sum to 1 for a given training instance.  I used a Sparse Categorical Cross-Entropy loss function because this is a classification problem and the dataset is sparse.  In order to validate each model, I'm using a 10% validation split.  I chose to train each model over 30 epochs.
## Results and Model Evaluation
After 30 epochs of training for each model, the models' validation accuracy continues to increase, and the validation loss continues to decrease.  This indicates to me that there is still room to improve my model.  Both models with 2 layers outperformed the 1 layer models.  However, 400 nodes vs 200 nodes per layer seemed to have minimal impact, when controlling for number of layers.  Future would could include adding more epochs of training, or adding more hidden layers.  Kaggle scores for my four models are: 0.96942, 0.97, 0.96157, 0.96228 (https://www.kaggle.com/brianelinsky Account User ID 4810027).  Additional stats for processing time, training set accuracy, and validation accuracy can be found in my lab notebook.
## Management Recommendations
Based off on my research so far, neural networks slightly outperform random forests and PCAs on the MNIST dataset.  If management considers a 97% accuracy rate is acceptable, this model can be deployed to production.  However, if higher accuracy scores are necessary, further research will be needed to increase accuracy to state of the art scores around 99.8% accuracy.