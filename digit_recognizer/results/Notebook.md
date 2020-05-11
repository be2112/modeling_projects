# Digit Recognizer Lab Notebook

## 2020-05-04

* Define the objective in business terms.
  * The objective is to classify as many digits correctly as possible.
* How will your solution be used?
  * This is not stated in the problem.
* What are the current solutions/workarounds (if any)?
  * Presumably the current solution is a manual one, and we are looking to automate it.
* How should you frame this problem (supervised/unsupervised, online/offline, etc.)?
  * This is an offline supervised learning problem.
* How should performance be measured?
  * Performance will be measured by categorization accuracy.
* Is the performance measure aligned with the business objective?
  * Not sure, as the business objective is not stated.
* What would be the minimum performance needed to reach the business objective?
  * I think above 90% would be pretty good.  And above 97% would be very good. http://yann.lecun.com/exdb/mnist/
* What are comparable problems?  Can you reuse experience or tools?
  * I haven't solved any comparable problems so far.
* Is human expertise available?
  * Yes, I am an expert (we all are) in handwriting recognition.
* How would you solve the problem manually?
  * I would solve this visually, not looking at the matrix of data.  This indicates to me that the current matrix may not be the best way to represent the data.  I may need to transform it before I train a model.
* List the assumptions you (or others) have made so far.
  * The data is labeled correctly.
  * The numbers are relatively centered.
* Verify assumptions if possible.



### Experiment Plan

This week I've been given explicit directions to do the following:

*Begin by fitting a random forest classifier using the full set of 784 explanatory variables and the model training set (train.csv). Record the time it takes to fit the model and then evaluate the model on the test.csv data by submitting to Kaggle.com. Provide your Kaggle.com score and user ID.*

### Experiment Results

RandomForestClassifier parameters:

```python
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': True, 'random_state': 98, 'verbose': 0, 'warm_start': False}
```

**OOB Score**: 0.9609761904761904

**Kaggle Score**: 0.96342 (Account User ID 4810027)(https://www.kaggle.com/brianelinsky/)

**Time**: 32.018 seconds (without parallelization)

## 2020-05-05

### Experiment Plan

My instructions are as follows:

*Execute principal components analysis (PCA) on the combined training and test set data together, generating principal components that represent 95 percent of the variability in the explanatory variables. The number of principal components in the solution should be substantially fewer than the 784 explanatory variables. Record the time it takes to identify the principal components.*

*Using the identified principal components, use the train.csv to build another random forest classifier. Record the time it takes to fit the model and to evaluate the model on the test.csv data by submitting to Kaggle.com. **Provide your Kaggle.com score and user ID.***

### Experiment Results

The PCA reduced the number of features to 332.



RandomForestClassifier parameters:

```python
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': True, 'random_state': 12, 'verbose': 0, 'warm_start': False}
0.9225238095238095
```

**OOB Score:**  0.9225238095238095

**Kaggle Score:** 0.93357 (Account User ID 4810027)(https://www.kaggle.com/brianelinsky/)

**Time:** 

* 11.166 seconds to train the PCA
* 73.140 seconds (without parallelization) to train the Random Forest Classifier

The issue with this experiment is the information leakage in the PCA.  The PCA was fit with both training data and test data.  Hence, the final model included some information from the test data.  This experiment should be replicated by fitting the PCA model with only training data, and using that model to transform the test data.  There was also information leakage in the scaler.

I was surprised that the OOB score and the Kaggle score were so close.  I expected the Kaggle score to be lower due to the information leakage.

## 2020-05-06

### Experiment Plan

Repeat the same experiment but without the information leakage.

*The experiment we have proposed has a MAJOR design flaw. Identify the flaw. Fix it. Rerun the experiment in a way that is consistent with a training-and-test regimen, and submit this to Kaggle.com. Provide your Kaggle.com score and user ID*

### Experiment Results

The PCA reduced the number of features to 320.

RandomForestClassifier parameters:

```python
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': True, 'random_state': 12, 'verbose': 0, 'warm_start': False}
```

**OOB Score:**  0.9203809523809524

**Kaggle Score:** 0.93114 (Account User ID 4810027)(https://www.kaggle.com/brianelinsky/)

**Time:** 

* 8.834 seconds to train the PCA
* 66.2 seconds (without parallelization) to train the Random Forest Classifier