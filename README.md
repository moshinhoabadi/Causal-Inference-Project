# Causal-Inference-Project

This is the repository that holds our project for "Introduction to Causal Inference" course, given by Dr. Uri Shalit and Mr. Rom Gutman.

In this project, we tried to determine the effect of holding the ball in the first half of a football match on the amount of shots on target in the second half of the match.


## The data

The data was mainly taken from two kaggle datasets:
1. European Soccer Database: https://www.kaggle.com/hugomathien/soccer
2. Supplementary Database: https://www.kaggle.com/jiezi2004/soccer


## The files

These are the code files we used:
- `EDA.ipynb`: We used this notebook for understanding the dataset before starting to work on the project. 
- `dataset_creation.ipynb`: We used this notebook for creating the required files from the raw dataset.
- `ATE_computations.py`:  We used this code for computing the ATE using several algorithms in several conditions.
- `main_analysis.ipynb`: We used this notebook for visualizing the results.


In addition, the repository contains the data directory that holds raw and transformed data, and the results directory that holds the ATE results for the several conditions we tried.



Note: the main database file is not found on this repository because it is too heavy. you can download from Kaggle and add to the data directory to run the repository by yourself.
