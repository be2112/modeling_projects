Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
25 May 2020


# Assignment 8: Language Modeling With an RNN

## Problem Description
The management team would like us to monitor customer reviews and complaint logs.  They would like to have a computer program identify the negative reviews so that a human can respond.  Presumably, humans are manually reviewing all of the reviews right now.  By adding this filtration step to the process, humans will be able to spend less time on the task.  If this model gets deployed to production, it will likely be an online model.  It will probably make predictions real-time.
## Data Preparation
I did more pre-processing for this project than for other ones.  I removed URLs, emojis, and punctuation from the tweets.  For each tweet, I split it up into a list of words.  Then I used GloVe to convert each word into a vector representation.
## Research Design and Modeling Methods
For the model, my first layer was an Embedding layer.  Se second layer as a SpatialDropout1D layer.  The third was a LSTM layer.  The last layer was a 1-node dense layer, with a sigmoid activation.  I used a binary cross entropy loss function because this is a binary classification problem.  And I used the accuracy metric.  Where my two models differed was the optimizer.  I fit one model with the Adam optimizer, and a second with the RMSprop optimizer.
## Results and Model Evaluation
Both models performed roughly the same.  The choice of optimizer didn't seem to make much of a difference.  The Adam optimizer version had a Kaggle score of 0.78527.  The RMSprop version had a Kaggle score of 0.78323.  (Account User ID 4810027)(https://www.kaggle.com/brianelinsky/).
## Management Recommendations
78% accuracy isn't a particularly high score from a business perspective.  Instead of optimizing accuracy, I could try optimizing for recall.  This would allow management to still filter down the total number of tweets for review, but also ensure that we don't miss any negative reviews.  I would recommend that instead of trying to increase the accuracy of the model.