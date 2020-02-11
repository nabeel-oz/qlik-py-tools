# Calling existing Machine Learning models

## Table of Contents

- [Introduction](#introduction)
- [Pre-requisites](#pre-requisites)
- [Setting up the model](setting-up-the-model)
- [Calling the model](calling-the-model)
    - [Building the features expression](building-the-features-expression)
    - [What-if analysis](what-if-analysis)
- [Additional functionality](additional-functionality)
- [Complete Example](complete-example)
- [Notes for Developers](notes-for-developers)
- [Attribution](#attribution)

## Introduction

This Server Side Extension (SSE) can be used to call existing Machine Learning (ML) models. Predictions can be obtained through the Qlik load script or through chart expressions. 

This capability enables you to deliver ML models in business focused Qlik apps. Predictions can be delivered within broader analysis and in the context of user selections. User controls in Qlik Sense can be used together with this capability to enable what-if analysis using the ML model.

This SSE also provides capabilities for training machine learning models entirely through the Qlik load script. This can be convenient for running experiments without writing any Python code. These capabilities are covered under the [Machine Learning](scikit-learn.md) and [Advanced Forecasting](Keras.md) sections.

## Pre-requisites

- This SSE currently supports scikit-learn and Keras models that have been saved to disk.
- The models need to be built with the same version of Python that is being used by the SSE (3.6.x). 
- scikit-learn models need to be saved using the Pickle library.
- Keras models need to be saved using the Keras model.save method.
- The Keras version needs to match the SSE.
- Preprocessing (e.g. scaling, OHE) needs to be handled by the model / pipeline.

## Setting up the model

This SSE will handle the communication between Qlik and Python and call the specified model. However, we need certain details for the model to translate the incoming data and call the model correctly. This information has to be supplied through a YAML file.

The YAML file needs to be placed in the SSE's `qlik-py-env/models` directory. The file needs to provide:

- **path**: Relative or absolute path to the model.
- **type**: Type of the model.
    - Currently supported values are `scikit-learn`, `sklearn` and `keras`.
- **preprocessor**: Optional preprocessor to prepare data for the model.
    - This has to be a path to a Python object that implements the `transform` method and has been saved using `Pickle`.
    - The SSE will call the preprocessor's `transform` method on the samples and use the output to call the model's prediction function.
- **features**: List of features expected by the model together with their data types.
    - The order of the features is important and needs to be followed by the model and the Qlik app.
    - The data types are required for correctly interpreting the data received from Qlik. Valid types are `int`, `float`, `str`, `bool`.
    - The names of the features should correspond to fields in the Qlik app. 

Here is a sample YAML file. You can also find complete examples [here](sample-scripts/HR-Attrition-v1.yaml) and [here](sample-scripts/HR-Attrition-v2.yaml).

```
---
path: ../pretrained/HR-Attrition-v1.pkl
type: sklearn
features:
    overtime : str
    salary : float
...
```

## Calling the model

A model can be called through a Qlik chart expression using the following syntax:

```
// PyTools.Predict('model-name', [FeaturesExpression], 'kwarg=value,...')
PyTools.Predict('HR-Attrition-v1', FeaturesExpression, 'debug=false')
```

- The model name is the YAML file name excluding the file extension. 
- The FeaturesExpression is a string concatenation of all the fields that will be passed as features to the model. Field values need to be separated by the `|` delimiter. There is a convenience function that can be used to build this expression which is covered below.
- The final argument is a string of key word arguments. Possible options are covered under [Additional functionality](additional-functionality).

The model can also be called through the Qlik load script. This requires an additional `Key` field that is used to identify each record and can be used to link the predictions back to the data model.

```
// Set up the information required to get predictions from the model:
// Model Name: Name of the YAML containing the model specifications. 
// Key: A field that can be used to link the data to the Qlik data model
// N_Features: A concatenated field providing input features for the model as specified in the YAML file.
// Kwargs: Additional key word arguments for the SSE.
TEMP_SAMPLES_WITH_KEYS:
LOAD
    'HR-Attrition-v1' as Model_Name,
    EmployeeNumber as Key,
    FeaturesExpression as N_Features,
    '' as Kwargs
RESIDENT [N_Features];

// Use the LOAD...EXTENSION syntax to call the Bulk_Predict function
[Predictions]:
LOAD
   model_name,
   key as EmployeeNumber,
   prediction
EXTENSION PyTools.Bulk_Predict(TEMP_SAMPLES_WITH_KEYS{Model_Name, Key, N_Features, Kwargs});

Drop table TEMP_SAMPLES_WITH_KEYS;
```

### Building the features expression
This SSE provides a convenience function to setup the features expression for Qlik based on the model's YAML file. This will only work if the feature names in the YAML file match the field names in Qlik.

This function can be called through the Qlik load script as shown below. You would do this prior to calling the model for predictions.

```
// Setup a temporary table with the model name.
// This should match the YAML file's name excluding the file extension.
TEMP_MODEL:
LOAD * INLINE [
    'Model_Name'
    'HR-Attrition-v1'
];

// Use a convenience function to get the features expression required by the model.
// This expression is based on the model specifications in the model's YAML file.
// The expression assumes that the field names in Qlik match the model specifications in the YAML file.
[FEATURES_EXPRESSION]:
LOAD
    result as features_expression
EXTENSION PyTools.Get_Features_Expression(TEMP_MODEL{Model_Name});

// Store the expression in a variable
vFeaturesExpression = peek('features_expression', 0, 'FEATURES_EXPRESSION');

Drop tables TEMP_MODEL, FEATURES_EXPRESSION;

// The Features Expression can be evaluated to obtain features concatenated into a single string as required by the SSE
[N_Features]:
LOAD
    EmployeeNumber,
    $(vFeaturesExpression) as FeaturesExpression
RESIDENT [HR-Data];
```

With this the model can be called through a chart expression as shown below.

```
PyTools.Predict('HR-Attrition-v1', FeaturesExpression, 'debug=false')
```

### What-if analysis

The features expression is simply a concatenation of fields in Qlik, and so we can use variables to override the value of fields to do what-if analysis. For example consider the partial expression below where a modifier is being applied to the `OverTime` and `PercentSalaryHike` features. 

```
...
    [NumCompaniesWorked] &'|'& 
    [Over18] &'|'& 
    //[OverTime] &'|'& 
    if('$(vNoOvertime)' = 'True', 'No', [OverTime]) &'|'& 
    //[PercentSalaryHike] &'|'& 
    $(vMonthlyIncrease) &'|'& 
    [PerformanceRating] &'|'& 
    [RelationshipSatisfaction] &'|'& 
...
```

The modifiers can then be exposed to the user through input controls such as buttons and sliders. 

The complete feature expression can be built manually by contatenating the requried fields, or by copying the value of the `vFeaturesExpression` variable setup by the convenience function described [above](building-the-features-expression).

### Additional functionality

### Complete Example

### Notes for Developers

## Attribution
The data used in the sample apps was obtained from https://www.kaggle.com:
- [Employee Attrition](https://www.kaggle.com/patelprashant/employee-attrition)