Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
1 May 2020


# Assignment 2: Evaluating Regression Models

## Problem Description
Given a set of explanatory variables for a house, my business objective is to predict its market value.  In this fictional scenario, I'm advising a real estate brokerage firm.  The firm currently uses conventional methods to price houses.  This model will complement those methods.  The model will be used to help ensure that the company buys houses below market value, and sells them above market value, ensuring a profit.

## Data Preparation
I dropped categorical variables that were missing over 10% of the data.  This included: 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'.  For numeric features, I imputed missing values with medians.  For categorical features, I imputed missing values with the most frequent value.  I encoded categorical features to dummy arrays.  I did not scale any variables.
## Research Design, Modeling Methods and Evaluation
My plan was to build random forest regressors with varying depths, then compare their test errors as the number of trees increases.  This would allow me to determine the optimal depth and number of trees.  A max depth of 20 outperformed 10, and was as good as higher depth numbers.  Next, I held depth constant at 20, but varied the number of features.  Unexpectedly, max_features=auto outperformed log2 and sqrt.
Using the same pre-processing pipeline, I next evaluated gradient boosting regressors.  The models with the lowest learning rates seemed to perform best.  It looked as if the test error would continue to decrease had I included more than 1000 trees in the forest.
Since many of the best models from my first set of experiments used parameters on the bounds of my grid search, I conducted a much larger grid search to select the final models.
The final random forest performed marginally better than my linear regression models from a few weeks ago.  However, the gradient boosting model performs substantially better.  Submitting both models to Kaggle resulted in a Log RMSE of 0.14935 for the random forest, and a Log RMSE of 0.12768 for the gradient boosting model (Account User ID 4810027)(https://www.kaggle.com/brianelinsky)
## Management Recommendations
I would recommend deploying the final version of the gradient boosting model to production.  I would advise management to not replace the conventional pricing methods with this model, rather to use the model to supplement the existing methods.  With a mean housing price of ~$180k, a model with an RMSE of ~$25k will give you a rough estimate of the housing price, but not a hyper-accurate one. Additional work could be done to improve the model.  That could include building an ensemble model of different regressors.

