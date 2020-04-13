Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
10 April 2020


# Assignment 1: Exploring and Visualizing Covid-19 Data

## Data Preparation
To prepare the data, I loaded it into a pandas dataframe.  From there I dropped the 'day,' 'month,' and 'year' fields because I felt those were redundant.  Next I formatted the 'dateRep' field as a Python datetime object.  Lastly, I renamed 'dateRep' to 'date'.

## Data Exploration

This dataset includes case, death, and population data for 204 countries.  The data set starts on 12/31/2019.  For many countries, data is not available for all of the days in the dataset.  This indicates that some countries are doing better at collecting data than others.  Additionally, it is a challenge to collect data from 200+ countries when there is no central international authority responsible for data collection.

As of April 1<sup>st</sup>, almost all countries have less than 10 deaths per day and 100 new cases per day.  But among large countries like the United States, Spain, France, and Germany, the amounts are much higher.  Graphing this data in bar chart format indicates that it the number of new cases and new deaths is likely exponentially distributed.

Graphing the daily case growth rate and daily death growth rate in the United States indicates that it is highly variable from day to day.  Additional work could be done to smooth the growth rate and remove a few outliers. 

When comparing new cases to new deaths in a scatterplot, there appears to be a weak positive relationship.  The relationship appears heteroscedastic.  As the number of cases increases, the numbers of deaths becomes more widely dispersed between countries.  This could indicate that countries are in different parts of the growth curve.  It could also indicate that some countries health care systems are being overrun, while others are able to handle the increased load.

When scaling cases and deaths using standard scaling and min max scaling, the relationship between cases and deaths becomes remarkably linear.
