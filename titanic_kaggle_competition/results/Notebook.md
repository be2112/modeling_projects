# Results Notebook

## 2020-04-04 Exploratory Analysis of the Data

### Findings

There are 891 observations.  Some data is missing for 'age' and 'cabin'.  The overall survival rate is 38%.  Mean age of all passengers is 30.  The Fare and SibSp variables exhibit significant positive skew.  The Fare and Pclass have the highest correlations with the survived variable.

#### Correlation Matrix

![2020-04-05_CorrMatrix](2020-04-04/CorrMatrix)

#### Histogram  

![2020-04-05_Histogram](2020-04-04/Histogram) 

### Interpretations

* Since Fare and Pclass have the highest correlations with survived, they will likely be the most useful variables in predicting who survives.  

### Next Steps

* We will need to decide what to do with missing age and cabin data.  Since so much of the cabin data is missing, we might need to throw that variable out.  For age we can likely start by imputing missing data with the median age.
* Will likely need to convert the Pclass and Survived variables from numeric to categorical variables.
* SibSp, Parch could either be modeled at categorical or numeric data.
* If I were to build a first model, I would likely want to try out a multiple regression model.