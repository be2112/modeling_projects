Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
18 April 2020


# Assignment 2: Evaluating Regression Models

## Problem Description
Given a set of explanatory variables for a house, my business objective is to predict its market value.  In this fictional scenario, I'm advising a real estate brokerage firm.  The firm currently uses conventional methods to price houses.  This model will complement those methods.  The model will be used to help ensure that the company buys houses below market value, and sells them above market value, ensuring a profit.
## Research Design and Modeling Methods
My plan was to start by fitting a basic linear regression model using a 10-fold cross validation methodology.  Then fit Lasso Regression, Ridge Regression, and Elastic Net Regression models all using the default hyperparameters and a 10-fold cross validation.  Then I would select the most promising model for fine tuning.  Fine tuning would involve a grid search over the hyperparameter space.  With the final model selected, I would then test it on my test dataset.
## Data Preparation
I dropped categorical variables that were missing over 10% of the data.  This included: 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'.  For numeric features, I imputed missing values with medians.  For categorical features, I imputed missing values with the most frequent value.  I encoded categorical features to dummy arrays.  Numeric variables were scaled using a robust scaler.

## Results and Model Evaluation
The average root mean square error of my basic linear model is $32,274.13.  This is an OK model, but definitely not great.  This is a good baseline to use to compare future models.
The ridge regression did perform slightly better than the standard linear regression, with a mean RMSE of $30,746.70.  The Lasso model had an average RMSE of $31,465.81.  The Elastic Net had an average RMSE of $33,778.98.  I decided to move forward with the Ridge Regression because it had the lowest RMSE.
I did a grid search over the ridge regression, varying the following parameters: alpha, normalize, max_iter, and tol.  Testing the best model on the test dataset resulted in a RMSE of $29,823.75.  The Kaggle Log RMSE was 0.15529 and I ranked in the 61st percentile.
## Management Recommendations
I would recommend deploying the final version of the Ridge Regression model to production.  I would advise management to not replace the conventional pricing methods with this model, rather to use the model to supplement the existing methods.  With a mean housing price of ~$180k, a model with an RMSE of ~$25k will give you a rough estimate of the housing price, but not a hyper-accurate one. Additional work could be done to improve the model.  That could include treating the pre-processing steps as hyperparamters, or fitting a different class of model, like a K-nearest neighbors model.

