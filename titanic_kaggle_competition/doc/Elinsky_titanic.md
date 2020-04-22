Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
26 April 2020


# Assignment 3: Evaluating Classification Models

## Problem Description
Given a set of explanatory variables for titanic passengers, my business objective is to predict whether or note each passenger will survive.  In this fictional scenario, I am providing evidence regarding characteristics associated survival to a historian writing a book.  This indicates to me that the model does not need to be extremely accurate at predicting survival on an individual basis.  Rather it is more important to develop a model that is interpretable, and predicts survival reasonably well.

## Research Design and Modeling Methods
My plan was to start by fitting a basic logistic regression model using a 10-fold cross validation methodology.  Then fit a simple Naive Bayes classifier.  Afterwards, I would select the more promising model, then fine tune it for the final predictions.

## Data Preparation
I decided to throw out the following variables: Name, Ticket, and Cabin.  I don't think Name or Ticket will have much predictive value.  There is a lot of missing data for the Cabin variable.  Missing categorical data was imputed with the most frequent values.  Missing numeric data was imputed with median values.

For the logistic regression, I encoded categorical variables as dummy variables.  For the Naive Bayes classifier, I encoded categorical variables as numbers with an Ordinal Encoder.

## Results and Model Evaluation
Both models performed similarly.  The AUC for the logistic regression was 0.849 and for the Bayes Naive classifier it was 0.831.  Accuracy for both was 79%.  This is a significant improvement over the 38% base rate of survival.

Next, I decided to add in regularization to my logistic regression model.  A grid search found that an L2 penalty with C=0.1 performed best, although the improvement was only slightly better than the simpler logistic regression with no penalty.

The learning curves indicated that this model likely under-fitting the training data.  The curves quickly reach a high plateau, and are very close to each other.  This indicates that a more complex model may improve its predictive capacity.

The Kaggle score for the final logistic regression model was 0.76555.  The kaggle score for the final Bayes Naive classifier was 0.75598 (https://www.kaggle.com/brianelinsky) (Account User ID 4810027).

## Management Recommendations
Examining the logistic regression coefficients indicates that survivors tended to be younger, richer, or female.  However, it is prudent to analyze these coefficients with a grain of salt.  The explanatory variables likely aren't independent of each other.  The predictive power of one attribute could be encoded in another.  A decision tree with a shallow depth would likely make a good enough model with better explainability.