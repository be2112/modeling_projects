Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
5 May 2020


# Assignment 3: Evaluating Classification Models

## Problem Description
Given an array of pixels and their associated degree of darkness, the objective is to classify each image as the correct digit.  As a manager, I need to decide if building a PCA model is worth the added time, or if my team should spend more time on the predictive model.
## Research Design and Modeling Methods
The experiment had us first fitting a random forest classifier.  Second, train a PCA on both the training and testing data.  Third, use the output of the PCA as features to train a new random forest classifier.  Finally, the objective was to correct the flaw in the original experiment design.
## Data Preparation
I did very little pre-processing.  The first random forest classifier had no pre-processing steps.  I utilized a StandardScaler for the PCA.
## Results and Model Evaluation
The simple random forest classifier, with primarily default hyperparameters, performed very well with a OOB score of 0.960976 and a similar Kaggle score of 0.96342 (Account User ID 4810027)(https://www.kaggle.com/brianelinsky/).  This model took 32 seconds to train, without parallelization.  The PCA reduced the number of features to 332, and took 11 seconds to train.  Feeding this into a random forest classifier produced an OOB score of 0.92252 and a Kaggle score of 0.93357.  It took 73 seconds to train this second random forest classifier.
The issue with this experiment is the information leakage in the PCA.  The PCA was fit with both training data and test data.  Hence, the final model included some information from the test data.  Next I replicated the experiment by fitting the PCA model with only training data, and using that model to transform the test data.  Surprisingly this model performed no better than the model with information leakage.  The OOB score was 0.92038, and the Kaggle score was 0.93114.  This PCA took slightly less time to train, at 9 seconds.  The random forest classifier was also slightly faster at 66 seconds.
## Management Recommendations
I would not recommend spending more time applying dimensionality reduction models on the MNIST dataset.  In some datasets, a significant proportion of the variance can be explained by the first 1-3 principal components.  When that is the case, you can gain some understanding of the dataset by visualizing the first few components.  However, I don't believe there is much business value in visualizing the first few components of a MNIST PCA.  Additionally, I didn't get much benefit from using PCA as a tool to extract features.