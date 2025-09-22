# Machine learning project: Fastest Lap Prediction from Practice Data
This project is part of **MOD550: Fundaments of Machine Learning for and with Engineering Applications** at the **University of Stavanger**. It uses machine learning techniques to predict the fastest lap times in a race based on practice session data. The goal is to explore how data from Free Practice sessions (FP1, FP2 and FP3) can be used to predict race performance.

## About me
My name is Simen S. Gåsland and I am currently studying **Computational Engineering** at the **University of Stavanger**. I am also a certified **Automation Technician** and holds a **Bachelor's degree in Automation Engineering**. 

## Tasks
### Task 1
* Make a histogram from a 2d random distribution.
* Make a 2d heat map from a 2d random distribution.
* Make a histogram for the source data you selected.
* Convert the histogram into a discrete PMF.
* Calculate the cumulative for each feature.

### Task 2
* Make a DataModel class that reads the output of the DataAquisition class (from task1) in its \_\_init\_\_()
* Make a function in DataModel to make a linear regression (using vanilla Python).
* Make a function that split the data you got from DataAquisition into train, validation and test (using vanilla Python).
* Make a function that computes MSE (from scrath).
* Make a function to make NN (using Keras).
* Make a function that does K_MEAN and GMM
* Make a linear regression on all your data (statistic).
* Make a linear regression on all your train data and test it on your validation.
* Compute the MSE on your validation data.
* Try for different distribution of initial data point
    - Discuss how different functions can be used in the linear regression, and different NN architecture. 
    - Discuss how you can use the validation data for the different cases. 
    - Discuss the different outcome from the different models when using the full dataset to train and when you use a different ML approach.
    - Discuss the outcomes you get for K-means and GMM.
    - Discuss how you can integrate supervised and unsupervised methods for your case.
