# Causal-Inference-Project

This is the repository that holds our project of the "Introduction to Causal Inference" course given by Dr. Uri Shalit and Mr. Rom Gutman.

In the project we tried to determine the affect of holding the ball in the first half of a football match on the amount of shots on target in the second half of the match.


## The data

The data was mainly taken from two kaggle datasets:
1. European Soccer Database: https://www.kaggle.com/hugomathien/soccer
2. Supplementary Database: https://www.kaggle.com/jiezi2004/soccer


## The files

There are some main files on this repository:
- dataset_creation.ipynb: is used to create the required files from the raw dataset.
- ATE_computations.py: is used to compute ATE values using several algorithms in several conditions.
- main_analysis.ipynb: is used to visualize the results of the ATE values.



note: the main database file from the kaggle is not found on this repository because its too heavy. you can download it and add to the data directory to run the repository by yourself.
