import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Workaround for Keras issue #1406
# "Using X backend." always printed to stdout #1406 
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras.models import Sequential
from keras.layers import Dense
sys.stderr = stderr

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Import a custom transformer for preprocessing data based on feature definitions
from preprocessor import Preprocessor

# Execute the script
if __name__ == '__main__':
    
    # Import feature definitions and data
    sheets = pd.read_excel('HR-Employee-Attrition.xlsx', sheet_name=["Feature Definitions", "Train-Test"])

    # Create feature definitions data frame
    features = sheets["Feature Definitions"]
    features.columns = [c.lower() for c in features.columns]
    features.set_index("name", append=False, inplace=True)

    # Setup the data dataframe
    data = sheets["Train-Test"]

    # Get the target features
    target = features.loc[features["variable_type"] == "target"]
    target_name = target.index[0]

    # Get the target data
    d_target = data.loc[:,[target_name]]

    # Get the features to be excluded from the model
    exclusions = features['variable_type'].isin(["excluded", "target", "identifier"])

    excluded = features.loc[exclusions]
    features = features.loc[~exclusions]

    # Remove excluded features from the data
    data = data[features.index.tolist()]

    # Split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(data, d_target, test_size=0.30, random_state=42)

    # Construct the ML pipelines
    pipe_lr = Pipeline([('prep', Preprocessor(features, return_type='df')), ('clf', LogisticRegression(solver='lbfgs', random_state=42))])
    pipe_rf = Pipeline([('prep', Preprocessor(features, return_type='df')), ('clf', RandomForestClassifier(n_estimators=10, random_state=42))])

    # List of pipelines for ease of iteration
    pipelines = [pipe_lr, pipe_rf]

    # Dictionary of pipelines and classifier types for ease of reference
    pipe_dict = {0: 'Logistic Regression', 1: 'Random Forest'}

    # Fit the pipelines
    for pipe in pipelines:
        pipe.fit(X_train, y_train.values.ravel())

    # Compare accuracies
    for idx, val in enumerate(pipelines):
        sys.stdout.write('\n%s pipeline train accuracy: %.3f\n' % (pipe_dict[idx], val.score(X_train, y_train)))
        sys.stdout.write('%s pipeline test accuracy: %.3f\n' % (pipe_dict[idx], val.score(X_test, y_test)))

    # Identify the most accurate model on test data
    best_acc = 0.0
    best_clf = 0
    best_pipe = ''
    for idx, val in enumerate(pipelines):
        if val.score(X_test, y_test) > best_acc:
            best_acc = val.score(X_test, y_test.values.ravel())
            best_pipe = val
            best_clf = idx
    sys.stdout.write('\nClassifier with best accuracy: %s\n' % pipe_dict[best_clf])

    # Save pipeline to file
    with open('HR-Attrition-v1.pkl', 'wb') as file:
        pickle.dump(best_pipe, file)
        sys.stdout.write('\nSaved %s pipeline to file\n\n' % pipe_dict[best_clf])

    # Also save the preprocessor and model as separate files
    with open('HR-Attrition-prep-v1.pkl', 'wb') as file:
        pickle.dump(best_pipe.named_steps['prep'], file)
    with open('HR-Attrition-clf-v1.pkl', 'wb') as file:
        pickle.dump(best_pipe.named_steps['clf'], file)
        
    # We will train a keras model as well
    # Run the training data through the preprocessor
    X_train_transformed = best_pipe.named_steps['prep'].transform(X_train)
    
    # Encode target values
    le = LabelEncoder().fit(y_train.values.ravel())
    y_train_encoded = le.transform(y_train.values.ravel())
    
    # Define the Keras model
    model = Sequential()
    model.add(Dense(100, input_dim=74, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the Keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the Keras model on the dataset
    model.fit(X_train_transformed, y_train_encoded, epochs=50, batch_size=8, class_weight={0:0.1, 1:2.0}, verbose=0)

    # Run the test data through the preprocessor
    X_test_transformed = best_pipe.named_steps['prep'].transform(X_test)
    # Encode the test labels
    y_test_encoded = le.transform(y_test.values.ravel())

    # Check model accuracy on test data
    sys.stdout.write('\nKeras test accuracy: %.3f\n\n' % (model.evaluate(X_test_transformed, y_test_encoded)[1]))

    # Save the keras model architecture and weights to disk
    model.save('HR-Attrition-Keras-v1.h5')

