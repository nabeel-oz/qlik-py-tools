# Supervised Machine Learning with scikit-learn

## Table of Contents

- [Introduction](#introduction)
- [Machine learning flow for Qlik](#machine-learning-flow-for-qlik)
- [Attribution](#attribution)

## Introduction

Supervised machine learning techniques make use of known samples to train a model, and then use this model to make predictions on new data. One of the best known machine learning libraries is [scikit-learn](http://scikit-learn.org/stable/index.html#), a package that provides efficient versions of a large number of well researched algorithms. A good introduction to machine learning and the scikit-learn API is available in [this excerpt from the Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.01-what-is-machine-learning.html). 

This SSE provides functions to train, test and evaluate models and then use these models to make predictions. The current implementation scope includes classification and regression algorithms.

## Machine learning flow for Qlik
Machine learning problems can be broken down into general steps. In this SSE each step is carried out with the help of functions that provide a bridge between Qlik and the scikit-learn API. These functions are explained in the sections below.

### Pre-requisites
1. The first step is to gather the features (i.e. dimensions and measures in Qlik) that will help us predict the target. The target can be a discrete label or class for a classification problem, or a continuous variable for a regression problem. 

   This is something where Qlik natively works well; bringing together data from multiple sources, deriving new dimensions and measures and structuring the data into a single table. This table should provide us one row for each sample, including the target and all the features being provided to predict the target.

   The first input to our model is the training dataset. If testing and evaluation needs to be done on the same dataset, we will split the data when setting up the model.

2. Next, for this implementation, we need to define the metadata for our dataset. This metadata can be brought into Qlik from any source such as a file or an internal table.

   For each feature, i.e. each column in the dataset, we need to define the following attributes:

| Metadata field | Description | Valid values | Remarks |
| --- | --- | --- | --- |
| Name | A unique name for the feature | Any string | The feature name must be unique. |
| Variable Type | Identify whether the variable is a feature or target | `feature`, `target`, `excluded`, `identifier` | |
| Data Type | Used to covert the data to the correct type | `bool`, `int`, `float`, `str` | |
| Feature Strategy | The feature preparation strategy | `one hot encoding`, `hashing`, `scaling`, `none` | Strings need to be converted to numerical values for machine learning. The strategies implemented in this SSE to do this are one hot encoding and hashing. Numerical values need to be scaled to avoid bias towards larger numbers. <br><br> In general, for discrete values use OHE where the unique values are small, otherwise use hashing. For continuous values, use scaling. |
| Hash Features | The number of features where the feature strategy is hashing | An integer e.g. `4` | The integer should be a power of 2 for the hashing to work correctly. |

### Setup the model

### Train and test the model

### Make predictions using the model

## Advanced topics
The basic flow needs to be extended in real world cases. 

### Optimize hyperparameters for a model
One key step is determining the best hyperparameters for an estimator. This process can be automated given a parameter grid and performing a search across combinations of the specified parameter values for the estimator.

This capability is currently a work in progress.

### Train multiple models 

### Out-of-core learning

### Dimensionality reduction

## Attribution
The data used in the sample apps was obtained from https://www.kaggle.com:
- [Employee Attrition](https://www.kaggle.com/patelprashant/employee-attrition)
