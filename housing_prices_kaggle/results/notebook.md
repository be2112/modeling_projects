# Housing Prices Kaggle Competition Notebook

## 2020-04-13

### Problem Framing

* Given a set of explanatory variables for a house, my business objective is to predict its market value.  In this fictional scenario, I'm advising a real estate brokerage firm.  The firm currently uses conventional methods to price houses.  This model will complement those methods.
* The model will be used to help ensure that the company buys houses below market value, and sells them above market value, ensuring a profit.
* Each row in the dataset represents a single housing transaction.  This is a supervised learning problem, because we are trying to predict price, and price is labeled in our dataset.  Regression models will be used, not classifiers, because the predicted variable is a number.
* My machine learning system will be a batch system.  I will train it using all of the available training data.  Then, if the results justify, we will deploy it to production.  It does not need to be an online model because the model does not need to learn on the fly.
* Performance of my model will be measured by the root mean squared error (RMSE) between the logarithm of the predicted price and the logarithm of the observed price.  Using the logarithm of the price  evenly penalizes errors on expensive and cheap houses.  This performance measurement aligns with the business objective:  consistent profit margin per house regardless of the sales price.
* I do not know the predictive accuracy required to meet the business objective.  I would need to know the effectiveness of the conventional pricing methods.  I would also want to know if the errors between my model and the conventional models are correlated.  If they are not correlated, then I have more room for error.
* This problem is similar to most pricing problems.
* I do not have access to any housing human expertise on this project.  If this were a real project, I could discuss the model with the people who employ the conventional pricing methods.
* If I had to solve this problem manually, I would probably create a linear regression in excel.  Or, I would estimate the value of a house to be the average price of its neighbors.  A K-Nearest Neighbors might also produce a good model.
* Assumptions I'm making:
  * The list of transactions is a representative sample of all of the transactions in Ames, Iowa.
  * Many of the features are qualitative.  I'm assuming that the assessors were calibrated.  That is, they were trained to provide consistent qualitative assessments.  e.g. Person A would have made the same qualitative assessment as Person B on the same house.

## 2020-04-15

### Experiment Plan

I plan to build a preprocessing pipeline using Scikit-Learn pipeline objects.  Then build a simple linear regression model and assess its accuracy with K-fold cross validation.

### Experimental Results

#### Preprocessing Decisions

* I dropped categorical variables that were missing over 10% of the data.  This included: 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'.
* For numeric features, I imputed missing values with medians.
* For categorical features, I imputed missing values with the most frequent value.
* I encoded categorical features to dummy arrays.
* Numeric variables were scaled using a robust scaler.
* A standard linear regression has no hyperparameters.

#### Results and Interpretation

| Metric             | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| RMSE Scores        | \$23,925.60, \$29,158.84, ​\$23,263.94, \$41,767.23,<br> \$29,711.50,\$43,453.35, \$24,383.14,   \$22,673.29,<br> \$61,916.91, $22,487.48 |
| Mean               | $32,274.13                                                   |
| Standard Deviation | $12,292.94                                                   |

![saleshist](https://raw.githubusercontent.com/be2112/modeling_projects/master/housing_prices_kaggle/results/2020-04-15/saleshist.png)

The average root mean square error of this basic linear model is $32,274.13.  Generally a lower RMSE is better than a higher RMSE.  This is an OK model, but definitely not great.  This is definitely a good baseline to compare future models to.  In practice, I do not think you would want to deploy a model with this poor of a fit.

## 2020-04-18

### Experiment Plan

I would like to build simple Ridge Regression, Lasso Regression, and Elastic Net models.  I will use default hyperparameters.  This should help me decide which model to choose for fine tuning.  I do not have many strong expectations, but I do think that adding some regularization should improve on the standard linear regression.

### Experimental Results

#### Preprocessing Decisions

* I decided to use the same pre-processing pipeline that I built for the linear regression.

#### Ridge Regression Results and Interpretation

| Metric             | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| Scores             | \$22341.07, \$24373.09, \$27642.86,\$38633.56<br> \$31945.70, \$27179.42, \$27325.23, \$25306.76<br> \$57088.56, $25630.72 |
| Mean               | $30,746.70                                                   |
| Standard Deviation | $9,791.03                                                    |

The ridge regression did perform slightly better than the standard linear regression.

#### Lasso Regression Results and Interpretation

| Metric             | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| Scores             | \$21131.29, ​\$25663.75, \$23090.21, \$41703.26,  \$29562.71, \$43187.42, \$24237.69, \$22631.90, \$61024.91, $22424.98 |
| Mean               | $31,465.81                                                   |
| Standard Deviation | $12,418.53                                                   |

When I fit this model with the default hyperparameters, I got a convergence warning.  It suggested I increased the number of iterations.  I did that, and got the same warning.  I suspect this is a product of the lasso regression.  In a Lasso regression, the gradients get smaller as you approach the global optimum.  I suspect that the model is bouncing around near the global optimum.

#### Elastic Net Regression Results and Interpretation

| Metric             | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| Scores             | \$21131.29, \$25663.75, \$23090.21, \$41703.26, \$29562.71,\$ 43187.42, \$24237.69, \$22631.90, \$61024.91, \$22424.98 |
| Mean               | $33,778.98                                                   |
| Standard Deviation | $10,492.77                                                   |

Surprisingly, the Elastic Net model performed the worst.  I plan on moving forward with the ridge regression and fine tuning that model.

#### Fine Tuning the Ridge Regression

I did a grid search over the ridge regression, varying the following parameters: alpha, normalize, 

 max_iter, and tol.  The best set of parameters were:

```python
Ridge(alpha=10.0, copy_X=True, fit_intercept=False, max_iter=10, normalize=True,
      random_state=None, solver='cholesky', tol=0.01)
```

This model produced the following scores on the training set:

| Metric             | Value                                                        |
| ------------------ | ------------------------------------------------------------ |
| Scores             | \$20824.86, \$24028.96, \$25201.07, \$38864.15, \$32577.72, \$25759.74, \$25427.44, \$23884.19, \$57740.97, \$23928.35 |
| Mean               | $29823.75                                                    |
| Standard Deviation | $10537.98                                                    |

And the following results on the Kaggle test set:

**Log RMSE**: 0.15529

**Percentile**: 61st

### Future Work

Additional work could include building an ensemble model, or attempting to model this dataset with a K-nearest neighbors model.  Additionally, it may be valuable to treat some of the pre-processing steps as hyperparemeters.
