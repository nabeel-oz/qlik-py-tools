# Supervised Machine Learning with scikit-learn

## Table of Contents

- [Introduction](#introduction)
- [Machine learning flow for Qlik](#machine-learning-flow-for-qlik)
     - [Preparing Data](#preparing-data)
     - [Preparing Feature Definitions](#preparing-feature-definitions)
     - [Setting up the model](#setting-up-the-model)
     - [Training and testing the model](#training-and-testing-the-model)
     - [Making predictions using the model](#making-predictions-using-the-model)
- [Advanced topics](#advanced-topics)
     - [Optimizing hyperparameters for a model](#optimizing-hyperparameters-for-a-model)
     - [Training multiple models](#training-multiple-models)
     - [Out-of-core learning](#out-of-core-learning)
     - [Dimensionality reduction](#dimensionality-reduction)
- [Attribution](#attribution)

## Introduction

Supervised machine learning techniques make use of known samples to train a model, and then use this model to make predictions on new data. One of the best known machine learning libraries is [scikit-learn](http://scikit-learn.org/stable/index.html#), a package that provides efficient versions of a large number of well researched algorithms. A good introduction to machine learning and the scikit-learn API is available in [this excerpt from the Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.01-what-is-machine-learning.html). 

This SSE provides functions to train, test and evaluate models and then use these models to make predictions. The current implementation scope includes classification and regression algorithms.

## Machine learning flow for Qlik
Machine learning problems can be broken down into general steps. In this SSE each step is carried out with the help of functions that provide a bridge between Qlik and the scikit-learn API. These functions are explained in the sections below.

At a high-level the steps are:
1. Prepare the training and testing dataset in Qlik
2. Prepare feature definitions in Qlik
3. Setup the model with relevant parameters
   - `PyTools.sklearn_Setup(model_name, estimator_args, scaler_args, execution_args)`
   - `PyTools.sklearn_Setup_Adv(model_name, estimator_args, scaler_args, metric_args, dim_reduction_args, execution_args)`
4. Optionally, setup a parameter grid to automate optimization of hyperparameters
   - `PyTools.sklearn_Param_Grid(model_name, estimator_args)` _(Work in progress)_
5. Set feature definitions for the model
   - `PyTools.sklearn_Set_Features(model_name, feature_name, variable_type, data_type, feature_strategy, hash_length)`
6. Fit the model using the training data, and optionally evalute it using test data
   - `PyTools.sklearn_Fit(model_name, n_features)`
   - `PyTools.sklearn_Partial_Fit(model_name, n_features)` _(Work in progress)_
7. Optionally, get metrics on test data if this was not done together with training
   - `PyTools.sklearn_Calculate_Metrics(model_name, n_features)`
   - `PyTools.sklearn_Get_Metrics(model_name)`
   - `PyTools.sklearn_Get_Confusion_Matrix(model_name, n_features)` _(Only applicable to classifiers)_
8. Get predictions from an existing model
   - `PyTools.sklearn_Predict(model_name, n_features)` _(For use in chart expressions)_
   - `PyTools.sklearn_Bulk_Predict(model_name, n_features)` _(For use in the load script)_
   - `PyTools.sklearn_Predict_Proba(model_name, n_features)` _(For use in chart expressions. Only applicable to classifiers)_
   - `PyTools.sklearn_Bulk_Predict_Proba(model_name, n_features)` _(For use in the load script. Only applicable to classifiers)_

Steps 1-7 are done through Qlik's data load processes, while the predictions can be made through either the load script or in real-time using chart expressions.

### Preparing Data
The first step is to gather the features (i.e. dimensions and measures in Qlik) that will help us predict the target. The target can be a discrete labels for a classification problem, or a continuous variable for a regression problem. 

This is something where Qlik natively works well; bringing together data from multiple sources, deriving new dimensions and measures and structuring the data into a single table. This table should provide us one row for each sample, including the target and all the features being provided to predict the target.

The first input to our model is the training dataset. If testing and evaluation needs to be done on the same dataset, we will split the data when setting up the model.
   
![training and testing data](images/sklearn-pre-01.png)

### Preparing Feature Definitions

Next, for this implementation, we need to provide feature definitions for our dataset. This metadata can be brought into Qlik from any source such as a file or an internal table.

For each feature, i.e. each column in the dataset, we need to define the following attributes:

| Metadata field | Description | Valid values | Remarks |
| --- | --- | --- | --- |
| Name | A unique name for the feature | Any string | The feature name must be unique. |
| Variable Type | Identify whether the variable is a feature or target | `feature`, `target`, `excluded`, `identifier` | |
| Data Type | Used to covert the data to the correct type | `bool`, `int`, `float`, `str` | |
| Feature Strategy | The feature preparation strategy | `one hot encoding`, `hashing`, `scaling`, `none` | Strings need to be converted to numerical values for machine learning. The strategies implemented in this SSE to do this are one hot encoding and hashing. Numerical values need to be scaled to avoid bias towards larger numbers. <br><br> In general, for discrete values use OHE where the unique values are small, otherwise use hashing. For continuous values, use scaling. |
| Hash Features | The number of features where the feature strategy is hashing | An integer e.g. `4` | The integer should be a power of 2 for the hashing to work correctly. |
   
The table should look like this:
![feature definitions](images/sklearn-pre-02.png)

### Setting up the model

### Training and testing the model

### Making predictions using the model

## Advanced topics
The basic flow needs to be extended in real world cases. 

### Optimizing hyperparameters for a model
One key step is determining the best hyperparameters for an estimator. This process can be automated given a parameter grid and performing a search across combinations of the specified parameter values for the estimator.

This capability is currently a work in progress.

### Training multiple models 

### Out-of-core learning

### Dimensionality reduction

## Attribution
The data used in the sample apps was obtained from https://www.kaggle.com:
- [Employee Attrition](https://www.kaggle.com/patelprashant/employee-attrition)
