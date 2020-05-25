Brian Elinsky  
Professor Lawrence Fulton, Ph.D.  
2020SP_MSDS_422-DL_SEC55  
20 May 2020


# Assignment 7: Image Processing with a CNN

## Problem Description
We are providing advice to a website provider that wants to automatically classify images.  The goal is the highest possible accuracy.  We are OK with sacrificing training time to get better accuracy.  The solution will be used to classify images that end users submit.
## Research Design and Modeling Methods
I decided to import an Xception model that was pretrained on the ImageNet dataset.  I used that CNN to extract features from the labeled pictures.  Then I fed those pictures into a small neural network designed to be a binary classifier.  The first layer was a dense layer, the second was a dropout layer, and the third was a one-node dense layer with sigmoid activation.
For the 2x2 factorial design, 2 models had the dropout layer, 2 did not.  I also varied the number of epochs, with 2 models using 30, and 2 models using 10. 
## Data Preparation
I did very little pre-processing.  I converted each image into a 3-dimensional array.  I created a layer for the red, another for the green, and a third blue component of the image.  Then I scaled the components to be between 0 and 1. 
## Results and Model Evaluation
Both models with dropout outperformed the models without dropout.  The effects of 10 vs 30 epochs of training were more mixed.  The best model had a validation accuracy of 98.8%.  The Kaggle Log Loss scores were: 0.40068, 0.35403, 0.42839, 0.42839 (Account User ID 4810027)(https://www.kaggle.com/brianelinsky/).
## Management Recommendations
With accuracy being a priority, using a pre-trained CNN is a must.  I would recommend moving forward with the pre-trained Xception model plus the neural network.  Additional work could be done to augment the image dataset.  That would increase the number of training examples, and it would make the model more robust.