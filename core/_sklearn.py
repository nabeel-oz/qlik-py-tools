import os
import gc
import sys
import ast
import time
import string
import locale
import pathlib
import warnings
import numpy as np
import pandas as pd
from tempfile import mkdtemp
from shutil import rmtree
from collections import OrderedDict
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, FastICA, NMF, SparsePCA,\
                                DictionaryLearning, LatentDirichletAllocation, MiniBatchDictionaryLearning, MiniBatchSparsePCA

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier,\
                            BaggingRegressor, ExtraTreesClassifier, ExtraTreesRegressor,\
                            GradientBoostingClassifier, GradientBoostingRegressor,\
                            RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lars, LarsCV,\
                                Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, LogisticRegression,\
                                LogisticRegressionCV, MultiTaskLasso, MultiTaskElasticNet, MultiTaskLassoCV, MultiTaskElasticNetCV,\
                                OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PassiveAggressiveClassifier,\
                                PassiveAggressiveRegressor, Perceptron, RANSACRegressor, Ridge, RidgeClassifier, RidgeCV,\
                                RidgeClassifierCV, SGDClassifier, SGDRegressor, TheilSenRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier,\
                            RadiusNeighborsRegressor
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier,\
                        ExtraTreeRegressor

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, KMeans,\
                            MiniBatchKMeans, MeanShift, SpectralClustering

from skater.model import InMemoryModel
from skater.core.explanations import Interpretation

# Workaround for Keras issue #1406
# "Using X backend." always printed to stdout #1406 
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras import backend as kerasbackend
sys.stderr = stderr

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import _utils as utils
from _machine_learning import Preprocessor, PersistentModel, TargetTransformer, Reshaper, KerasClassifierForQlik, KerasRegressorForQlik
import ServerSideExtension_pb2 as SSE

# Add Generated folder to module path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

class SKLearnForQlik:
    """
    A class to implement scikit-learn classification and regression algorithmns for Qlik.
    http://scikit-learn.org/stable/modules/classes.html#api-reference
    """
    
    # Counter used to name log files for instances of the class
    log_no = 0
    
    # Ordered Dictionary to cache recently used models at the class level
    model_cache = OrderedDict()
    
    # Limit on the number of models to be cached
    cache_limit = 3
    
    def __init__(self, request, context, path="../models/"):
        """
        Class initializer.
        :param request: an iterable sequence of RowData
        :param context:
        :param path: a directory path to store persistent models
        :Sets up the model parameters based on the request
        """
               
        # Set the request, context and path variables for this object instance
        self.request = request
        self.context = context
        self.path = path
        self.logfile = None
        
        # Set up a dictionary of valid algorithmns
        self.algorithms = {"DummyClassifier":DummyClassifier, "DummyRegressor":DummyRegressor,\
                           "AdaBoostClassifier":AdaBoostClassifier, "AdaBoostRegressor":AdaBoostRegressor,\
                           "BaggingClassifier":BaggingClassifier, "BaggingRegressor":BaggingRegressor,\
                           "ExtraTreesClassifier":ExtraTreesClassifier, "ExtraTreesRegressor":ExtraTreesRegressor,\
                           "GradientBoostingClassifier":GradientBoostingClassifier,\
                           "GradientBoostingRegressor":GradientBoostingRegressor,\
                           "RandomForestClassifier":RandomForestClassifier, "RandomForestRegressor":RandomForestRegressor,\
                           "VotingClassifier":VotingClassifier, "GaussianProcessClassifier":GaussianProcessClassifier,\
                           "GaussianProcessRegressor":GaussianProcessRegressor, "LinearRegression":LinearRegression,\
                           "LogisticRegression":LogisticRegression, "LogisticRegressionCV":LogisticRegressionCV,\
                           "PassiveAggressiveClassifier":PassiveAggressiveClassifier,\
                           "PassiveAggressiveRegressor":PassiveAggressiveRegressor, "Perceptron":Perceptron,\
                           "RANSACRegressor":RANSACRegressor, "Ridge":Ridge, "RidgeClassifier":RidgeClassifier,\
                           "RidgeCV":RidgeCV, "RidgeClassifierCV":RidgeClassifierCV, "SGDClassifier":SGDClassifier,\
                           "SGDRegressor":SGDRegressor, "TheilSenRegressor":TheilSenRegressor, "BernoulliNB":BernoulliNB,\
                           "GaussianNB":GaussianNB, "MultinomialNB":MultinomialNB,\
                           "KNeighborsClassifier":KNeighborsClassifier, "KNeighborsRegressor":KNeighborsRegressor,\
                           "RadiusNeighborsClassifier":RadiusNeighborsClassifier,\
                           "RadiusNeighborsRegressor":RadiusNeighborsRegressor, "MLPClassifier":MLPClassifier,\
                           "MLPRegressor":MLPRegressor, "LinearSVC":LinearSVC, "LinearSVR":LinearSVR, "NuSVC":NuSVC,\
                           "NuSVR":NuSVR, "SVC":SVC, "SVR":SVR, "DecisionTreeClassifier":DecisionTreeClassifier,\
                           "DecisionTreeRegressor":DecisionTreeRegressor, "ExtraTreeClassifier":ExtraTreeClassifier,\
                           "ExtraTreeRegressor":ExtraTreeRegressor, "PCA":PCA, "KernelPCA":KernelPCA, "IncrementalPCA":IncrementalPCA,\
                           "TruncatedSVD":TruncatedSVD, "FactorAnalysis":FactorAnalysis, "FastICA":FastICA, "NMF":NMF,\
                           "SparsePCA":SparsePCA, "DictionaryLearning":DictionaryLearning,\
                           "LatentDirichletAllocation":LatentDirichletAllocation,\
                           "MiniBatchDictionaryLearning":MiniBatchDictionaryLearning, "MiniBatchSparsePCA":MiniBatchSparsePCA,\
                           "AffinityPropagation":AffinityPropagation, "AgglomerativeClustering":AgglomerativeClustering,\
                           "Birch":Birch, "DBSCAN":DBSCAN, "FeatureAgglomeration":FeatureAgglomeration, "KMeans":KMeans,\
                            "MiniBatchKMeans":MiniBatchKMeans, "MeanShift":MeanShift, "SpectralClustering":SpectralClustering,\
                            "ARDRegression":ARDRegression, "BayesianRidge":BayesianRidge, "ElasticNet":ElasticNet,\
                            "ElasticNetCV":ElasticNetCV, "HuberRegressor":HuberRegressor, "Lars":Lars, "LarsCV":LarsCV,\
                            "Lasso":Lasso, "LassoCV":LassoCV, "LassoLars":LassoLars, "LassoLarsCV":LassoLarsCV, "LassoLarsIC":LassoLarsIC,\
                            "MultiTaskLasso":MultiTaskLasso, "MultiTaskElasticNet":MultiTaskElasticNet, "MultiTaskLassoCV":MultiTaskLassoCV,\
                            "MultiTaskElasticNetCV":MultiTaskElasticNetCV, "OrthogonalMatchingPursuit":OrthogonalMatchingPursuit,\
                            "OrthogonalMatchingPursuitCV":OrthogonalMatchingPursuitCV, "KerasRegressor":KerasRegressorForQlik,\
                            "KerasClassifier":KerasClassifierForQlik}
        
        self.decomposers = {"PCA":PCA, "KernelPCA":KernelPCA, "IncrementalPCA":IncrementalPCA, "TruncatedSVD":TruncatedSVD,\
                            "FactorAnalysis":FactorAnalysis, "FastICA":FastICA, "NMF":NMF, "SparsePCA":SparsePCA,\
                            "DictionaryLearning":DictionaryLearning, "LatentDirichletAllocation":LatentDirichletAllocation,\
                            "MiniBatchDictionaryLearning":MiniBatchDictionaryLearning, "MiniBatchSparsePCA":MiniBatchSparsePCA}
        
        self.classifiers = ["DummyClassifier", "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",\
                            "GradientBoostingClassifier", "RandomForestClassifier", "VotingClassifier",\
                            "GaussianProcessClassifier", "LogisticRegression", "LogisticRegressionCV",\
                            "PassiveAggressiveClassifier", "Perceptron", "RidgeClassifier", "RidgeClassifierCV",\
                            "SGDClassifier", "BernoulliNB", "GaussianNB", "MultinomialNB", "KNeighborsClassifier",\
                            "RadiusNeighborsClassifier", "MLPClassifier", "LinearSVC", "NuSVC", "SVC",\
                            "DecisionTreeClassifier", "ExtraTreeClassifier", "KerasClassifier"] 
        
        self.regressors = ["DummyRegressor", "AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor",\
                           "GradientBoostingRegressor", "RandomForestRegressor", "GaussianProcessRegressor",\
                           "LinearRegression", "PassiveAggressiveRegressor", "RANSACRegressor", "Ridge", "RidgeCV",\
                           "SGDRegressor", "TheilSenRegressor", "KNeighborsRegressor", "RadiusNeighborsRegressor",\
                           "MLPRegressor", "LinearSVR", "NuSVR", "SVR", "DecisionTreeRegressor",\
                           "ExtraTreeRegressor", "ARDRegression", "BayesianRidge", "ElasticNet", "ElasticNetCV",\
                           "HuberRegressor", "Lars", "LarsCV", "Lasso", "LassoCV", "LassoLars", "LassoLarsCV",\
                           "LassoLarsIC", "MultiTaskLasso", "MultiTaskElasticNet", "MultiTaskLassoCV", "MultiTaskElasticNetCV",\
                           "OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV", "KerasRegressor"]
    
        self.clusterers = ["AffinityPropagation", "AgglomerativeClustering", "Birch", "DBSCAN", "FeatureAgglomeration", "KMeans",\
                           "MiniBatchKMeans", "MeanShift", "SpectralClustering"]

    def list_models(self):
        """
        List available models.
        This function is only meant to be used as a chart expression in Qlik.
        """
        
        # Interpret the request data based on the expected row and column structure
        row_template = ['strData']
        col_headers = ['search_pattern']
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
        
        # Get the list of models based on the search pattern
        search_pattern = self.request_df.loc[0, 'search_pattern']
        
        # If the search pattern is empty default to all models
        if not search_pattern.strip():
            search_pattern = '*'
        
        # Get the list of models as a string
        models = "\n".join([str(p).split("\\")[-1] for p in list(pathlib.Path(self.path).glob(search_pattern))])
        
        # Prepare the output
        self.response = pd.Series(models)
        
        # Finally send the response
        return self.response
    
    def setup(self, advanced=False):
        """
        Initialize the model with given parameters
        Arguments are retreived from the keyword argument columns in the request data
        Arguments should take the form of a comma separated string: 'arg1=value1, arg2=value2'
        For estimater, scaler and dimensionality reduction hyperparameters the type should also be specified
        Use the pipe | character to specify type: 'arg1=value1|str, arg2=value2|int, arg3=value3|bool' 
        """
        
        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'strData', 'strData', 'strData']
        col_headers = ['model_name', 'estimator_args', 'scaler_args', 'execution_args']
        
        if advanced:
            # If specified, get dimensionality reduction arguments
            row_template = ['strData', 'strData', 'strData', 'strData', 'strData', 'strData']
            col_headers = ['model_name', 'estimator_args', 'scaler_args', 'metric_args', 'dim_reduction_args',\
                           'execution_args']
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
               
        # Create a model that can be persisted to disk
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.loc[0, 'model_name']
        
        # Get the argument strings from the request dataframe
        estimator_args = self.request_df.loc[0, 'estimator_args']
        scaler_args = self.request_df.loc[0, 'scaler_args']
        execution_args = self.request_df.loc[0, 'execution_args']
        if advanced:
            metric_args = self.request_df.loc[0, 'metric_args']
            dim_reduction_args = self.request_df.loc[0, 'dim_reduction_args']
            
            if len(dim_reduction_args) > 0:
                self.model.dim_reduction = True
            else:
                self.model.dim_reduction = False 
        
            # Set the relevant parameters using the argument strings
            self._set_params(estimator_args, scaler_args, execution_args, metric_args=metric_args,\
                             dim_reduction_args=dim_reduction_args)
        else:
            # Set the relevant parameters using the argument strings
            self._set_params(estimator_args, scaler_args, execution_args)
            self.model.dim_reduction = False 
        
        # Persist the model to disk
        self.model = self.model.save(self.model.name, self.path, overwrite=self.model.overwrite, compress=self.model.compress)
        
        # Update the cache to keep this model in memory
        self._update_cache()
              
        # Prepare the output
        message = [[self.model.name, 'Model successfully saved to disk',\
                    time.strftime('%X %x %Z', time.localtime(self.model.state_timestamp))]]
        self.response = pd.DataFrame(message, columns=['model_name', 'result', 'time_stamp'])
        
        # Send the reponse table description to Qlik
        self._send_table_description("setup")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def set_param_grid(self):
        """
        Set a parameter grid that will be used to optimize hyperparameters for the estimator.
        The parameters are used in the fit method to do a grid search.
        """

        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'strData', 'strData']
        col_headers = ['model_name', 'estimator_args', 'grid_search_args']
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
       
        # Initialize the persistent model
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.loc[0, 'model_name']
        
        # Get the estimator's hyperparameter grid from the request dataframe
        param_grid = self.request_df.loc[:, 'estimator_args']

        # Get the grid search arguments from the request dataframe
        grid_search_args = self.request_df.loc[0, 'grid_search_args']

        # Get the model from cache or disk
        self._get_model()
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(3)

        self._set_grid_params(param_grid, grid_search_args)
        
        # Persist the model to disk
        self.model = self.model.save(self.model.name, self.path, overwrite=self.model.overwrite, compress=self.model.compress)
        
        # Update the cache to keep this model in memory
        self._update_cache()
              
        # Prepare the output
        message = [[self.model.name, 'Hyperparameter grid successfully saved to disk',\
                    time.strftime('%X %x %Z', time.localtime(self.model.state_timestamp))]]
        self.response = pd.DataFrame(message, columns=['model_name', 'result', 'time_stamp'])
        
        # Send the reponse table description to Qlik
        self._send_table_description("setup")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def keras_setup(self):
        """
        Setup the architecture for a Keras model.
        This function should be called after a model has been initialized with the setup (and optionally set_param_grid) methods.

        The architecture should define the layers, optimization and compilation parameters for a Keras sequential model.

        Since this SSE handles preprocessing the number of features will be inferenced from the training data.
        However, when using 3D or 4D data the input_shape needs to be specified in the first layer. 
        Note that this SSE does not support the keras input_dim argument. Please always use input_shape.

        The request should contain: model name, sort order, layer type, args, kwargs
        With a row for each layer, with a final row for compilation keyword arguments.
        
        Arguments should take the form of a comma separated string with the type of the value specified.
        Use the pipe | character to specify type: 'arg1=value1|str, arg2=value2|int, arg3=value3|bool'

        Sample input expected from the request:
        'DNN', 1, 'Dense', '12|int', 'activation=relu|str'
        'DNN', 2, 'Dropout', '0.25|float', ''
        'DNN', 3, 'Dense', '1|int', 'activation=sigmoid|str'
        'DNN', 4, 'Compilation', '', 'loss=binary_crossentropy|str, optimizer=adam|str, metrics=accuracy|list|str'

        If you want to specify parameters for the optimizer, you can add that as the second last row of the architecture:
        ...
        'DNN', 4, 'SGD', '', 'lr=0.01|float, clipvalue=0.5|float'
        'DNN', 5, 'Compilation', '', 'loss=binary_crossentropy|str, metrics=accuracy|str'     

        Layer wrappers can be used by including them in the layer type followed by a space:
        'DNN', 1, 'TimeDistributed Dense' ...
        ...
        For the Bidirectional wrapper the merge_mode parameter can be specified in the kwargs.  

        When using recurrent or convolutional layers you will need to pass a valid input_shape to reshape the data.
        Additionally, the feature definitions should include an 'identifier' variable that will be used for reshaping.
        E.g. The identifier is a date field and you want to use a LSTM with 10 time steps on a dataset with 20 features.
        'RNN', 1, 'LSTM', '64|int', 'input_shape=10;20|tuple|int, activation=relu|str'
        Note that you can pass None instead of 20 for the features, as the actual number of features will be calculated after 
        preprocessing the data.

        """

        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'numData', 'strData', 'strData', 'strData']
        col_headers = ['model_name', 'sort_order', 'layer_type', 'args', 'kwargs']
                
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
               
        # Create a model that can be persisted to disk
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.loc[0, 'model_name']
        
        # Get the model from cache or disk
        self._get_model()
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(3)

        # Set a flag which will let us know that this is a Keras model
        self.model.using_keras = True

        # Sort the layers, drop unnecessart columns and save the model architecture to a new data frame
        architecture = self.request_df.sort_values(by=['sort_order']).reset_index(drop=True).drop(labels=['model_name', 'sort_order'], axis = 1)

        # Convert args to a list and kwargs to a dictionary
        architecture['args'] = architecture['args'].apply(utils.get_args_by_type)
        architecture['kwargs'] = architecture['kwargs'].apply(utils.get_kwargs).apply(utils.get_kwargs_by_type) 

        # Add the architecture to the model
        self.model.architecture = architecture

        # Add the first layer's kwargs as a property for easy reference
        self.model.first_layer_kwargs = self.model.architecture.iloc[0, 2]

        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(10)

        # Persist the model to disk
        self.model = self.model.save(self.model.name, self.path, overwrite=self.model.overwrite, compress=self.model.compress)
        
        # Update the cache to keep this model in memory
        self._update_cache()
              
        # Prepare the output
        message = [[self.model.name, 'Keras model architecture saved to disk',\
                    time.strftime('%X %x %Z', time.localtime(self.model.state_timestamp))]]
        self.response = pd.DataFrame(message, columns=['model_name', 'result', 'time_stamp'])
        
        # Send the reponse table description to Qlik
        self._send_table_description("setup")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response

    def set_features(self):
        """
        Add feature definitions for the model
        """
        
        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'strData', 'strData', 'strData', 'strData', 'strData']
        col_headers = ['model_name', 'name', 'variable_type', 'data_type', 'feature_strategy', 'strategy_args']
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
       
        # Initialize the persistent model
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.loc[0, 'model_name']
        
        # Get the model from cache or disk
        self._get_model()
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(3)
        
        # Add the feature definitions to the model
        self.model.features_df = self.request_df
        self.model.features_df.set_index("name", drop=False, inplace=True)
        # Store a copy of the features_df that will remain untouched in later calls
        self.model.original_features_df = self.model.features_df.copy()

        # Ensure there is at most one feature with variable_type identifier
        if len(self.model.features_df.loc[self.model.features_df["variable_type"] == "identifier"]) > 1:
            err = "Invalid feature definitions. Detected more than one feature with variable_type set to identifier. You can only pass one unique identifier."
            raise Exception(err)

        # Persist the model to disk
        self.model = self.model.save(self.model.name, self.path, overwrite=self.model.overwrite, compress=self.model.compress)
        
        # Update the cache to keep this model in memory
        self._update_cache()
        
        # Prepare the output
        message = [[self.model.name, 'Feature definitions successfully saved to model',\
                    time.strftime('%X %x %Z', time.localtime(self.model.state_timestamp))]]
        self.response = pd.DataFrame(message, columns=['model_name', 'result', 'time_stamp'])
        
        # Send the reponse table description to Qlik
        self._send_table_description("setup")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def get_features(self):
        """
        Get feature definitions for an existing model
        """
        
        # Get the model from cache or disk based on the model_name in request
        self._get_model_by_name()
        
        # Prepare the output
        self.response = self.model.features_df
        self.response["sort_order"] = pd.Series([i+1 for i in range(len(self.response.index))], index=self.response.index)
        self.response = self.response[["model_name", "sort_order", "name", "variable_type", "data_type",\
                                       "feature_strategy", "strategy_args"]]
        
        # Send the reponse table description to Qlik
        self._send_table_description("features")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def fit(self):
        """
        Train and test the model based on the provided dataset
        """
        
        # Open an existing model and get the training & test dataset and targets
        train_test_df, target_df = self._get_model_and_data(target=True, set_feature_def=True)
        
        # Check that the estimator is an supervised ML algorithm
        if self.model.estimator_type not in ["classifier", "regressor"]:
            err = "Incorrect usage. The estimator specified is not a known classifier or regressor: {0}".format(self.model.estimator)
            raise Exception(err)
        
        # Check which validation strategy is to be used, if any
        # For an explanation of cross validation in scikit-learn see: http://scikit-learn.org/stable/modules/cross_validation.html#multimetric-cross-validation
        if self.model.time_series_split > 0:
            self.model.validation = "timeseries"
            # Set up cross validation to be performed using TimeSeriesSplit
            self.model.cv = TimeSeriesSplit(n_splits=self.model.time_series_split, max_train_size=self.model.max_train_size)
        elif self.model.cv > 0:
            self.model.validation = "k-fold"
        elif self.model.test_size > 0:
            self.model.validation = "hold-out"
        else:
            self.model.validation = "external"

        if self.model.validation == "hold-out":        
            # Split the data into training and testing subsets
            self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(train_test_df, target_df, test_size=self.model.test_size, random_state=self.model.random_state)
        else:
            self.X_train = train_test_df
            self.y_train = target_df
        
        # Add the training and test data to the model if required
        if self.model.retain_data:
            self.model.X_train = self.X_train
            self.model.y_train = self.y_train
            
            try:
                self.model.X_test = self.X_test
                self.model.y_test = self.y_test
            except AttributeError:
                pass
        
        # Scale the targets and increase stationarity if required
        if self.model.scale_target or self.model.make_stationary:
            # Set up the target transformer
            self.model.target_transformer = TargetTransformer(scale=self.model.scale_target, make_stationary=self.model.make_stationary, stationarity_lags=self.model.stationarity_lags,\
                missing=self.model.missing, scaler=self.model.scaler, logfile=self.logfile, **self.model.scaler_kwargs)

            # Fit the transformer to the training targets
            self.model.target_transformer = self.model.target_transformer.fit(self.y_train)

            # Apply the transformer to the training targets
            self.y_train = self.model.target_transformer.transform(self.y_train)
            # Drop samples where the target cannot be transformed due to insufficient lags
            self.X_train = self.X_train.iloc[len(self.X_train)-len(self.y_train):] 
        
        # Add lag observations to the samples if required
        if self.model.lags or self.model.lag_target:
            # Check if the current sample will be included as an input, or whether we only use lag observations for predictions
            extrapolate = 1 if self.model.current_sample_as_input else 0
            # Add the lag observations
            self.X_train = self._add_lags(self.X_train, self.y_train, extrapolate=extrapolate, update_features_df=True)
            # Drop targets for samples which were dropped due to null values after adding lags.
            if len(self.y_train) > len(self.X_train):
                self.y_train = self.y_train.iloc[len(self.y_train)-len(self.X_train):]

        # If this is a Keras estimator, we require the preprocessing to return a data frame instead of a numpy array
        prep_return = 'df' if self.model.using_keras else 'np'

        # Construct the preprocessor
        prep = Preprocessor(self.model.features_df, return_type=prep_return, scale_hashed=self.model.scale_hashed, scale_vectors=self.model.scale_vectors,\
        missing=self.model.missing, scaler=self.model.scaler, logfile=self.logfile, **self.model.scaler_kwargs)

        # Setup a list to store steps for the sklearn pipeline
        pipe_steps = [('preprocessor', prep)]

        if self.model.dim_reduction:
            # Construct the dimensionality reduction object
            reduction = self.decomposers[self.model.reduction](**self.model.dim_reduction_args)
            
            # Include dimensionality reduction in the pipeline steps
            pipe_steps.append(('reduction', reduction))
            self.model.estimation_step = 2
        else:
            self.model.estimation_step = 1      

        # If this is a Keras estimator, update the input shape and reshape the data if required
        if self.model.using_keras:
            # Update the input shape based on the final number of features after preprocessing
            self._keras_update_shape(prep)

            # Add the Keras build function, architecture and prediction_periods to the estimator keyword arguments
            self.model.estimator_kwargs['build_fn'] = self._keras_build_fn
            self.model.estimator_kwargs['architecture'] = self.model.architecture
            self.model.estimator_kwargs['prediction_periods'] = self.model.prediction_periods

            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(10)
            
            # Check than an identifier has been provided for sorting data if this is a sequence prediction problem
            if self.model.lags or len(self.model.first_layer_kwargs["input_shape"]) > 1:
                assert len(self.model.original_features_df[self.model.original_features_df['variable_type'].isin(["identifier"])]) == 1, \
                    "An identifier is mandatory when using lags or with sequence prediction problems. Define this field in your feature definitions."

            # Cater for multi-step predictions
            if self.model.prediction_periods > 1:
                # Transform y to a vector of values equal to prediction_periods
                self.y_train = utils.vectorize_array(self.y_train, steps=self.model.prediction_periods)
                # Drop values from x for which we don't have sufficient y values
                self.X_train = self.X_train.iloc[:-len(self.X_train)+len(self.y_train)]

            # Add a pipeline step to update the input shape and reshape the data if required
            # This transform will also add lag observations if specified through the lags parameter
            # If lag_target is True, an additional feature will be created for each sample using the previous value of y 
            reshape = Reshaper(first_layer_kwargs=self.model.first_layer_kwargs, logfile=self.logfile)
            pipe_steps.append(('reshape', reshape))
            self.model.estimation_step += self.model.estimation_step

            # Avoid tensorflow error for keras models
            # https://github.com/tensorflow/tensorflow/issues/14356
            # https://stackoverflow.com/questions/40785224/tensorflow-cannot-interpret-feed-dict-key-as-tensor
            kerasbackend.clear_session()
        
        # Try assuming the pipeline involves a grid search
        try:
            # Construct an estimator
            estimator = self.algorithms[self.model.estimator](**self.model.estimator_kwargs)

            # Prepare the grid search using the previously set parameter grid
            grid_search = GridSearchCV(estimator=estimator, param_grid=self.model.param_grid, **self.model.grid_search_args)
            
            # Add grid search to the pipeline steps
            pipe_steps.append(('grid_search', grid_search))

            # Construct the sklearn pipeline using the list of steps
            self.model.pipe = Pipeline(pipe_steps)

            if self.model.validation in ["k-fold", "timeseries"]:
                # Perform K-fold cross validation
                self._cross_validate()

            # Fit the training data to the pipeline
            if self.model.using_keras:
                # https://stackoverflow.com/questions/54652536/keras-tensorflow-backend-error-tensor-input-10-specified-in-either-feed-de
                session = tf.Session()
                kerasbackend.set_session(session)
                with session.as_default():
                    with session.graph.as_default():
                        sys.stdout.write("\nMODEL: {}, INPUT SHAPE: {}\n\n".format(self.model.name, self.model.first_layer_kwargs['input_shape']))
                        y = self.y_train.values if self.y_train.shape[1] > 1 else self.y_train.values.ravel()
                        self.model.pipe.fit(self.X_train, y)
            else:
                self.model.pipe.fit(self.X_train, self.y_train.values.ravel())

            # Get the best parameters and the cross validation results
            grid_search = self.model.pipe.named_steps['grid_search']
            self.model.best_params = grid_search.best_params_
            self.model.cv_results = grid_search.cv_results_

            # Get the best estimator to add to the final pipeline
            estimator = grid_search.best_estimator_

            # Update the pipeline with the best estimator
            self.model.pipe.steps[self.model.estimation_step] = ('estimator', estimator)

        except AttributeError:
            # Construct an estimator
            estimator = self.algorithms[self.model.estimator](**self.model.estimator_kwargs)

            # Add the estimator to the pipeline steps
            pipe_steps.append(('estimator', estimator))

            # Construct the sklearn pipeline using the list of steps
            self.model.pipe = Pipeline(pipe_steps)

            if self.model.validation in ["k-fold", "timeseries"]:
                # Perform K-fold cross validation
                self._cross_validate()

            # Fit the training data to the pipeline
            if self.model.using_keras:
                # https://stackoverflow.com/questions/54652536/keras-tensorflow-backend-error-tensor-input-10-specified-in-either-feed-de
                session = tf.Session()
                kerasbackend.set_session(session)
                with session.as_default():
                    with session.graph.as_default():
                        sys.stdout.write("\nMODEL: {}, INPUT SHAPE: {}\n\n".format(self.model.name, self.model.first_layer_kwargs['input_shape']))
                        y = self.y_train.values if self.y_train.shape[1] > 1 else self.y_train.values.ravel()
                        self.model.pipe.fit(self.X_train, y)
            else:
                self.model.pipe.fit(self.X_train, self.y_train.values.ravel())
        
        if self.model.validation == "hold-out":       
            # Evaluate the model using the test data            
            self.calculate_metrics(caller="internal")
        
        if self.model.calc_feature_importances:
            # Select the dataset for calculating importances
            if self.model.validation == "hold-out":
                X = self.X_test
                y = self.y_test # Already a numpy array after calculate_metrics
            else:
                X = self.X_train
                y = self.y_train.values.ravel()
            
            # Calculate model agnostic feature importances
            self._calc_importances(X = X, y = y)

        # Persist the model to disk
        self.model = self.model.save(self.model.name, self.path, overwrite=self.model.overwrite, compress=self.model.compress)
                
        # Update the cache to keep this model in memory
        self._update_cache()
        
        # Prepare the output
        if self.model.validation != "external": 
            message = [[self.model.name, 'Model successfully trained, tested and saved to disk.',\
                        time.strftime('%X %x %Z', time.localtime(self.model.state_timestamp)),\
                        "{0} model has a score of {1:.3f} against the test data."\
                        .format(self.model.estimator, self.model.score), self.model.score]]
        else:
            message = [[self.model.name, 'Model successfully trained and saved to disk.',\
                        time.strftime('%X %x %Z', time.localtime(self.model.state_timestamp)),\
                        "{0} model score unknown as test_size was <= 0."\
                        .format(self.model.estimator), np.NaN]]
            
        self.response = pd.DataFrame(message, columns=['model_name', 'result', 'time_stamp', 'score_result', 'score'])
        
        # Send the reponse table description to Qlik
        self._send_table_description("fit")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
        
    # TO DO : Allow for larger datasets by using partial fitting methods avaialble with some sklearn algorithmns
    # def partial_fit(self):
        
    def fit_transform(self, load_script=False):
        """
        Fit the data to the model and then transform.
        This method is meant to be used for unsupervised learning models for clustering and dimensionality reduction.
        The models can be fit and tranformed through the load script or through chart expressions in Qlik.
        The load_script flag needs to be set accordingly for the correct response.
        """

        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'strData']
        col_headers = ['model_name', 'n_features']
        feature_col_num = 1
        
        # An additional key field column is expected if the call is made through the load script
        if load_script:
            row_template = ['strData', 'strData', 'strData']
            col_headers = ['model_name', 'key', 'n_features']
            feature_col_num = 2
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
               
        # Initialize the persistent model
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.loc[0, 'model_name']
        
        # Get the model from cache or disk
        self._get_model()
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(3)
        
        # Check that the estimator is an unsupervised ML algorithm
        if self.model.estimator_type not in ["decomposer", "clusterer"]:
            err = "Incorrect usage. The estimator specified is not a known decompostion or clustering algorithm: {0}".format(self.model.estimator)
            raise Exception(err)

        if load_script:
            # Set the key column as the index
            self.request_df.set_index("key", drop=False, inplace=True)

        # Split the features provided as a string into individual columns
        self.X = pd.DataFrame([x[feature_col_num].split("|") for x in self.request_df.values.tolist()], columns=self.model.features_df.loc[:,"name"].tolist(),\
        index=self.request_df.index)
        
        # Convert the data types based on feature definitions 
        self.X = utils.convert_types(self.X, self.model.features_df)

        # Construct the preprocessor
        prep = Preprocessor(self.model.features_df, scale_hashed=self.model.scale_hashed, scale_vectors=self.model.scale_vectors,\
        missing=self.model.missing, scaler=self.model.scaler, logfile=self.logfile, **self.model.scaler_kwargs)
        
        # Create a chache for the pipeline's transformers
        # https://scikit-learn.org/stable/modules/compose.html#caching-transformers-avoid-repeated-computation
        # cachedir = mkdtemp()

        # Construct a sklearn pipeline
        self.model.pipe = Pipeline([('preprocessor', prep)]) #, memory=cachedir)

        if self.model.dim_reduction:
            # Construct the dimensionality reduction object
            reduction = self.decomposers[self.model.reduction](**self.model.dim_reduction_args)
            
            # Include dimensionality reduction in the sklearn pipeline
            self.model.pipe.steps.insert(1, ('reduction', reduction))
            self.model.estimation_step = 2
        else:
            self.model.estimation_step = 1  

        # Construct an estimator
        estimator = self.algorithms[self.model.estimator](**self.model.estimator_kwargs)

        # Add the estimator to the sklearn pipeline
        self.model.pipe.steps.append(('estimator', estimator))  

        # Fit the data to the pipeline
        if self.model.estimator_type == "decomposer":
            # If the estimator is a decomposer we apply the fit_transform method at the end of the pipeline
            self.y = self.model.pipe.fit_transform(self.X)

            # Prepare the response
            self.response = pd.DataFrame(self.y, index=self.X.index)

        elif self.model.estimator_type == "clusterer":
            # If the estimator is a decomposer we apply the fit_predict method at the end of the pipeline
            self.y = self.model.pipe.fit_predict(self.X)

            # Prepare the response
            self.response = pd.DataFrame(self.y, columns=["result"], index=self.X.index)
                
        # Clear the cache directory setup for the pipeline's transformers
        # rmtree(cachedir)
        
        # Update the cache to keep this model in memory
        self._update_cache()
        
        if load_script:
            # Add the key field column to the response
            self.response = self.request_df.join(self.response).drop(['n_features'], axis=1)
        
            # If the function was called through the load script we return a Data Frame
            if self.model.estimator_type == "decomposer":
                self._send_table_description("reduce")
            elif self.model.estimator_type == "clusterer":
                self._send_table_description("cluster")
            
            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(4)
            
            return self.response
            
        # If the function was called through a chart expression we return a Series
        else:
            # Dimensionality reduction is only possible through the load script
            if self.model.estimator_type == "decomposer":
                err = "Dimensionality reduction is only possible through the load script."
                raise Exception(err)
            
            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(4)
            
            return self.response.loc[:,'result']
    
    def calculate_metrics(self, caller="external", ordered_data=False): 
        """
        Return key metrics based on a test dataset.
        Metrics returned for a classifier are: accuracy, precision, recall, fscore, support
        Metrics returned for a regressor are: r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score
        """
        
        # If the function call was made externally, process the request
        if caller == "external":
            # Open an existing model and get the training & test dataset and targets based on the request
            self.X_test, self.y_test = self._get_model_and_data(target=True, ordered_data=ordered_data)    

        # Keep a copy of the y_test before any transformations
        y_test_copy = self.y_test.copy()
        # Calculate if any extra lag periods are expected when making the target stationary using differencing
        extra_lags = max(self.model.stationarity_lags) if self.model.lag_target and self.model.make_stationary=='difference' else 0

        # Scale the targets and increase stationarity if required
        if self.model.scale_target or self.model.make_stationary:
            # If using differencing, we assume sufficient lag values for inversing the transformation later
            y_lags = self.y_test.iloc[:extra_lags].values if self.model.make_stationary=='difference' else None
            # Apply the transformer to the test targets
            self.y_test = self.model.target_transformer.transform(self.y_test)  
            # Drop samples where self.y_test cannot be transformed due to insufficient lags
            self.X_test = self.X_test.iloc[len(self.X_test)-len(self.y_test):]

        # Refresh the keras model to avoid tensorflow errors
        if self.model.using_keras:
            self._keras_refresh()
        
        # Get predictions based on the samples
        if ordered_data:
            self.y_pred = self.sequence_predict(variant="internal")
           
            # Handle possible null values where a prediction could not be generated
            self.y_pred = self.y_pred[self.placeholders:]
            self.y_test = y_test_copy.iloc[self.placeholders+extra_lags:]

            # Inverse transformations predictions if required 
            if self.model.scale_target or self.model.make_stationary:
                # Apply the transformer to the test targets
                self.y_pred = self.y_pred if y_lags is None else np.append(y_lags, self.y_pred)
                self.y_pred = self.model.target_transformer.inverse_transform(self.y_pred) 
                # Remove lags used for making the series stationary in case of differencing
                if self.model.make_stationary == 'difference':
                    self.y_pred = self.y_pred[extra_lags:]
        else:
            self.y_pred = self.model.pipe.predict(self.X_test)

            # Inverse transformations on the predictions if required
            if self.model.scale_target or self.model.make_stationary:
                # Apply the transformer to the predictions
                self.y_pred = self.model.target_transformer.inverse_transform(self.y_pred)
                # Reset y_test to orginal values
                self.y_test = y_test_copy
        
        # Flatten the y_test DataFrame
        self.y_test = self.y_test.values.ravel()
        
        # Try getting the metric_args from the model
        try:
            metric_args = self.model.metric_args
        except AttributeError: 
            metric_args = {}
               
        if self.model.estimator_type == "classifier":
            labels = self.model.pipe.named_steps['estimator'].classes_
            
            # Check if the average parameter is specified
            if len(metric_args) > 0  and "average" in metric_args:
                # Metrics are returned as an overall average
                metric_rows = ["overall"]
            else:
                # Get the class labels to be used as rows for the result DataFrame
                metric_rows = labels
            
            # Get key classifier metrics
            metrics_df = pd.DataFrame([x for x in metrics.precision_recall_fscore_support\
                                       (self.y_test, self.y_pred, **metric_args)],\
                                      index=["precision", "recall", "fscore", "support"], columns=metric_rows).transpose()
            # Add accuracy
            self.model.score = metrics.accuracy_score(self.y_test, self.y_pred)
            metrics_df.loc["overall", "accuracy"] = self.model.score
            # Finalize the structure of the result DataFrame
            metrics_df.loc[:,"model_name"] = self.model.name
            metrics_df.loc[:,"class"] = metrics_df.index
            metrics_df = metrics_df.loc[:,["model_name", "class", "accuracy", "precision", "recall", "fscore", "support"]]
            
            # Prepare the confusion matrix and add it to the model
            self._prep_confusion_matrix(self.y_test, self.y_pred, labels)
            
        elif self.model.estimator_type == "regressor":
            # Get the r2 score
            self.model.score = metrics.r2_score(self.y_test, self.y_pred, **metric_args)
            metrics_df = pd.DataFrame([[self.model.score]], columns=["r2_score"])
                        
            # Get the mean squared error
            metrics_df.loc[:,"mean_squared_error"] = metrics.mean_squared_error(self.y_test, self.y_pred, **metric_args)
            
            # Get the mean absolute error
            metrics_df.loc[:,"mean_absolute_error"] = metrics.mean_absolute_error(self.y_test, self.y_pred, **metric_args)
            
            # Get the median absolute error
            metrics_df.loc[:,"median_absolute_error"] = metrics.median_absolute_error(self.y_test, self.y_pred)
            
            # Get the explained variance score
            metrics_df.loc[:,"explained_variance_score"] = metrics.explained_variance_score(self.y_test, self.y_pred, **metric_args)

            # If the target was scaled we need to inverse transform certain metrics to the original scale
            # However, if we used the sequence prediction function, the inverse transform has already been performed
            if not ordered_data and (self.model.scale_target or self.model.make_stationary):
                for m in ["mean_squared_error", "mean_absolute_error", "median_absolute_error"]:
                    metrics_df.loc[:, m] = self.model.target_transformer.inverse_transform(metrics_df.loc[:, [m]], array_like=False).values.ravel()
            
            # Finalize the structure of the result DataFrame
            metrics_df.loc[:,"model_name"] = self.model.name
            metrics_df = metrics_df.loc[:,["model_name", "r2_score", "mean_squared_error", "mean_absolute_error",\
                                           "median_absolute_error", "explained_variance_score"]]
           
        if caller == "external":
            
            if self.model.calc_feature_importances:
                # Calculate model agnostic feature importances
                self._calc_importances(X = self.X_test, y = self.y_test)

            self.response = metrics_df

            # Send the reponse table description to Qlik
            if self.model.estimator_type == "classifier":
                self._send_table_description("metrics_clf")
            elif self.model.estimator_type == "regressor":
                self._send_table_description("metrics_reg")

            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(4)

            # Finally send the response
            return self.response
        else:
            # Save the metrics_df to the model
            self.model.metrics_df = metrics_df
        
    def get_confusion_matrix(self):
        """
        Returns a confusion matrix calculated previously using testing data with fit or calculate_metrics
        """
        
        # Get the model from cache or disk based on the model_name in request
        self._get_model_by_name()
        
        try:
            # Prepare the output
            self.response = self.model.confusion_matrix
        except AttributeError:
            err = "The confusion matrix is only avaialble for classifiers, and when hold-out testing " + \
            "or K-fold cross validation has been performed."
            raise Exception(err)
        
        # Send the reponse table description to Qlik
        if "step" in self.response.columns:
            self._send_table_description("confusion_matrix_multi")
        else:
            self._send_table_description("confusion_matrix")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def predict(self, load_script=False, variant="predict"):
        """
        Return a prediction by applying an existing model to the supplied data.
        If variant='predict_proba', return the predicted probabilties for each sample. Only applicable for certain classes.
        If variant='predict_log_proba', return the log probabilities for each sample. Only applicable for certain classes.
        This method can be called from a chart expression or the load script in Qlik.
        The load_script flag needs to be set accordingly for the correct response.
        """
        
        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'strData']
        col_headers = ['model_name', 'n_features']
        feature_col_num = 1
        
        # An additional key field column is expected if the call is made through the load script
        if load_script:
            row_template = ['strData', 'strData', 'strData']
            col_headers = ['model_name', 'key', 'n_features']
            feature_col_num = 2
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
               
        # Initialize the persistent model
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.loc[0, 'model_name']
        
        # Get the model from cache or disk
        self._get_model()
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(3)
        
        if load_script:
            # Set the key column as the index
            self.request_df.set_index("key", drop=False, inplace=True)
        
        try:
            # Split the features provided as a string into individual columns
            self.X = pd.DataFrame([x[feature_col_num].split("|") for x in self.request_df.values.tolist()],\
                                        columns=self.model.features_df.loc[:,"name"].tolist(),\
                                        index=self.request_df.index)
        except AssertionError as ae:
            err = "The number of input columns do not match feature definitions. Ensure you are using the | delimiter and that the target is not included in your input to the prediction function."
            raise AssertionError(err) from ae
        
        # Convert the data types based on feature definitions 
        self.X = utils.convert_types(self.X, self.model.features_df, sort=False)

        if variant in ('predict_proba', 'predict_log_proba'):
            # If probabilities need to be returned
            if variant == 'predict_proba':
                # Get the predicted probability for each sample 
                self.y = self.model.pipe.predict_proba(self.X)
            elif variant == 'predict_log_proba':
                # Get the log probability for each sample
                self.y = self.model.pipe.predict_log_proba(self.X)
                
            # Prepare a list of probability by class for each sample
            probabilities = []

            for a in self.y:
                s = ""
                i = 0
                for b in a:
                    s = s + ", {0}: {1:.3f}".format(self.model.pipe.named_steps['estimator'].classes_[i], b)
                    i = i + 1
                probabilities.append(s[2:])
            
            self.y = probabilities
                
        else:
            # Predict y for X using the previously fit pipeline
            self.y = self.model.pipe.predict(self.X)

            # Inverse transformations on the targets if required
            if self.model.scale_target or self.model.make_stationary:
                # Apply the transformer to the test targets
                self.y = self.model.target_transformer.inverse_transform(self.y) 

        # Prepare the response
        self.response = pd.DataFrame(self.y, columns=["result"], index=self.X.index)
        
        if load_script:
            # Add the key field column to the response
            self.response = self.request_df.join(self.response).drop(['n_features'], axis=1)
        
            # If the function was called through the load script we return a Data Frame
            self._send_table_description("predict")
            
            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(4)
            
            return self.response
            
        # If the function was called through a chart expression we return a Series
        else:
            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(4)
            
            return self.response.loc[:,'result']
    
    def sequence_predict(self, load_script=False, variant="predict"):
        """
        Make sequential predictions using a trained model. 
        This function is built for sequence and time series predictions. For standard ML simply use the predict function.
        For sequence prediction we expect using lag periods as features. 
        So previous samples, and possibly predictions, need to be fed as input for the next prediction. 
        Therefore, ensure that the number of features and historical data passed to get a prediction matches the model's requirements.
        
        If the previous target is being included in the lag observations, i.e. the lag_target parameter was set to True,
        include historical target values in n_features in the same order as the feature definitions provided during model training. 
        The target can be empty or 0 for future periods.
        For multi-step predictions, the features can be empty or 0 for future periods as well.

        If variant='predict_proba', we return the predicted probabilties for each sample. Otherwise, we return the predictions.
        If variant='internal', the call can be made from within this class.
        
        This method can be called from a chart expression or the load script in Qlik. 
        The load_script flag needs to be set accordingly for the correct response.
        """

        if variant != 'internal':
            # Open an existing model and get the input dataset. 
            # Target for historical data are expected if using previous targets as a feature.
            request_data = self._get_model_and_data(ordered_data=True) 
            if type(request_data) == list:
                X, y = request_data
            else:
                X = request_data
        else:
            X = self.X_test.copy()
            y = self.y_test.copy()

        # Scale the targets and increase stationarity if required
        if variant != 'internal' and self.model.lag_target and (self.model.scale_target or self.model.make_stationary):
            # If using differencing, we assume sufficient lag values for inversing the transformation later
            y_lags = y.iloc[:max(self.model.stationarity_lags)].values if self.model.make_stationary=='difference' else None
            # Apply the transformer to the targets
            y = self.model.target_transformer.transform(y)
            # Drop samples where y cannot be transformed due to insufficient lags
            X = X.iloc[len(X)-len(y):]

        # Set the number of periods to be predicted
        prediction_periods = self.model.prediction_periods
        # Set the number of rows required for one prediction
        rows_per_pred = 1
        extra_lags = max(self.model.stationarity_lags) if self.model.lag_target and self.model.make_stationary=='difference' else 0

        # Check that the input data includes history to meet any lag calculation requirements
        if self.model.lags:
            # Check if the current sample will be included as an input, or whether we only use lag observations for predictions
            extrapolate = 1 if self.model.current_sample_as_input else 0        
            # An additional lag observation is needed if previous targets are being added to the features
            rows_per_pred = self.model.lags+extrapolate+1 if self.model.lag_target else self.model.lags+extrapolate
            # If the target is being lagged and made stationary through differencing additional lag periods are required
            if self.model.lag_target and self.model.make_stationary=='difference':
                extra_msg = " plus an additional {} periods for making the target stationary using differencing".format(extra_lags)
            # For multi-step predictions we only expect lag values, not the current period's values
            # rows_per_pred = rows_per_pred-1 if prediction_periods > 1 else rows_per_pred
            assert len(X) >= rows_per_pred + extra_lags, "Insufficient input data as the model requires {} lag periods for each prediction".format(rows_per_pred) + extra_msg

        if variant != 'internal':
            # Prepare the response DataFrame
            # Initially set up with the 'model_name' and 'key' columns and the same index as request_df
            self.response = self.request_df.drop(columns=['n_features'])
        
        # Set up a list to contain predictions and probabilities if required
        predictions = []
        get_proba =  False
        if variant == 'predict_proba':
            get_proba =  True
            probabilities = []     

        # Refresh the keras model to avoid tensorflow errors
        if self.model.using_keras:
            self._keras_refresh()

        if prediction_periods > 1:
            if not self.model.lag_target:
                y = None

            # Check that we can generate 1 or more predictions of prediction_periods each
            n_samples = len(X)
            assert (n_samples - rows_per_pred) >= prediction_periods, \
                "Cannot generate predictions for {} periods with {} rows, with {} rows required for lag observations. You may need to provide more historical data or sufficient placeholder rows for future periods."\
                .format(prediction_periods, n_samples, rows_per_pred)
            
            # For multi-step predictions we can add lag observations up front as we only use actual values
            # i.e. We don't use predicted y values for further predictions    
            if self.model.lags or self.model.lag_target:
                X = self._add_lags(X, y=y, extrapolate=extrapolate)   

            # We start generating predictions from the first row as lags will already have been added to each sample
            start = 0
        else:
            # We start generating predictions from the point where we will have sufficient lag observations
            start = rows_per_pred
        
        if self.model.lag_target or prediction_periods > 1:
            # Get the predictions by walking forward over the data
            for i in range(start, len(X), prediction_periods):      
                # For multi-step predictions we take in rows_per_pred rows of X to generate predictions for prediction_periods
                if prediction_periods > 1:
                    batch_X = X.iloc[[i]]
                    
                    if not get_proba:
                        # Get the prediction. 
                        pred = self.model.pipe.predict(batch_X)
                        # Flatten the predictions for multi-step outputs and add to the list
                        pred = pred.ravel().tolist()
                        predictions += pred
                    else:
                        # Get the predicted probability for each sample 
                        proba = self.model.pipe.predict_proba(batch_X)
                        proba = proba.reshape(-1, len(self.model.pipe.named_steps['estimator'].classes_))
                        probabilities += proba.tolist()
                # For walk forward predictions with lag targets we use each prediction as input to the next prediction, with X values avaialble for future periods.
                else:
                    batch_X = X.iloc[i-rows_per_pred : i] 
                    # Add lag observations
                    batch_y = y.iloc[i-rows_per_pred : i]
                    batch_X = self._add_lags(batch_X, y=batch_y, extrapolate=extrapolate)

                    # Get the prediction. We only get a prediction for the last sample in the batch, the remaining samples only being used to add lags.
                    pred = self.model.pipe.predict(batch_X.iloc[[-1],:])

                    # Add the prediction to the list. 
                    predictions.append(pred)
                                
                    # Add the prediction to y to be used as a lag target for the next prediction
                    y.iloc[i, 0] = pred

                    # If probabilities need to be returned
                    if get_proba:
                        # Get the predicted probability for each sample 
                        probabilities.append(self.model.pipe.predict_proba(batch_X.iloc[[-1],:]))
        else:
            # Add lag observations to the samples if required
            if self.model.lags:
                X = self._add_lags(X, extrapolate=extrapolate)

            # Get prediction for X
            predictions = self.model.pipe.predict(X)

            # If probabilities need to be returned
            if get_proba:
                # Get the predicted probability for each sample 
                probabilities = self.model.pipe.predict_proba(X)
        
        # Set the number of placeholders needed in the response
        # These are samples for which predictions were not generated due to insufficient lag periods or for meeting multi-step prediction period requirements
        self.placeholders = rows_per_pred + extra_lags
        # Transform probabilities to a readable string
        if get_proba:
            # Add the required number of placeholders at the start of the response list
            y = ["\x00"] * self.placeholders
            
            # Truncate multi-step predictions if the (number of samples - rows_per_pred) is not a multiple of prediction_periods
            if prediction_periods > 1 and ((n_samples-rows_per_pred) % prediction_periods) > 0:              
                probabilities = probabilities[:-len(probabilities)+(n_samples-rows_per_pred)]
            
            for a in probabilities:
                s = ""
                i = 0
                for b in a:
                    s = s + ", {0}: {1:.3f}".format(self.model.pipe.named_steps['estimator'].classes_[i], b)
                    i += 1
                y.append(s[2:])

        # Prepare predictions
        else:
            if prediction_periods > 1:
                # Set the value to use for nulls
                null = np.NaN if is_numeric_dtype(np.array(predictions)) else "\x00"

                # Truncate multi-step predictions if the (number of samples - rows_per_pred) is not a multiple of prediction_periods
                if (n_samples-rows_per_pred) % prediction_periods > 0:
                    predictions = predictions[:-len(predictions)+(n_samples-rows_per_pred)]
                
                # Add null values at the start of the response list to match the cardinality of the input from Qlik
                y = np.array(([null] * self.placeholders) + predictions)
            elif self.model.lag_target: 
                # Remove actual values for which we did not generate predictions due to insufficient lags
                if is_numeric_dtype(y.iloc[:, 0].dtype):
                    y.iloc[:self.placeholders, 0] = np.NaN
                else:
                    y.iloc[:self.placeholders, 0] = "\x00"
                # Flatten y to the expected 1D shape
                y = y.values.ravel()
            else:
                y = np.array(predictions)
            
            # Inverse transformations on the targets if required  
            if variant != 'internal' and (self.model.scale_target or self.model.make_stationary):
                # Apply the transformer to the test targets
                placeholders = y[:self.placeholders] if prediction_periods > 1 or self.model.lag_target else []
                y = y if y_lags is None else np.append(y_lags, y[len(placeholders):])
                y = self.model.target_transformer.inverse_transform(y) 
                
                # Replace lags used for making the series stationary with nulls in case of differencing
                if self.model.make_stationary == 'difference':
                    null = np.NaN if is_numeric_dtype(np.array(predictions)) else "\x00"
                    y = np.append(np.array([null]*extra_lags), y[extra_lags:])
                
                # Add back the placeholders for lag values
                if len(placeholders) > 0:
                    y = np.append(placeholders, y)
        if variant == 'internal':
            return y

        # Add predictions / probabilities to the response
        self.response['result'] = y

        # Reindex the response to reset to the original sort order
        self.response = self.response.reindex(self.original_index)
        
        if load_script:
            # If the function was called through the load script we return a Data Frame
            self._send_table_description("predict")
            
            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(4)
            
            return self.response
            
        # If the function was called through a chart expression we return a Series
        else:
            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(4)
            
            return self.response.loc[:,'result']    
    
    def explain_importances(self):
        """
        Explain feature importances for the requested model
        """

        # Get the model from cache or disk based on the model_name in request
        self._get_model_by_name()

        # Get the feature importances calculated in the calculate_metrics method
        try:
            self.response = self.model.importances
        except AttributeError:
            err = "Feature importances are not available. Check that the execution argument calculate_importances " +\
            "is set to True, and that test_size > 0 or the Calculate_Metrics function has been executed."
            raise Exception(err) 

        # Add the model name to the response and rearrange columns
        self.response.loc[:, "model_name"] = self.model.name
        self.response = self.response[["model_name", "feature_name", "importance"]]

        # Send the reponse table description to Qlik
        self._send_table_description("importances")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def get_features_expression(self):
        """
        Get a string that can be evaluated in Qlik to get the features portion of the predict function
        """
        
        # Get the model from cache or disk based on the model_name in request
        self._get_model_by_name()
        
        # Prepare the expression as a string
        delimiter = " &'|'& "

        # Get the complete feature definitions for this model
        features_df = self.model.original_features_df.copy()
    
        # Set features that are not expected in the features expression in Qlik
        exclude = ["excluded"]

        if not self.model.lag_target:
            exclude.append("target")
        if not self.model.lags:
            exclude.append("identifier")

        # Exclude columns that are not expected in the request data
        exclusions = features_df['variable_type'].isin(exclude)
        features_df = features_df.loc[~exclusions]
    
        # Get the feature names
        features = features_df["name"].tolist()
        
        # Prepare a string which can be evaluated to an expression in Qlik with features as field names
        self.response = pd.Series(delimiter.join(["[" + f + "]" for f in features]))
        
        # Send the reponse table description to Qlik
        self._send_table_description("expression")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response

    def get_best_params(self):
        """
        Get the best parameters for the model based on the grid search cross validation 
        """
        
        # Get the model from cache or disk based on the model_name in request
        self._get_model_by_name()
        
        try:
            # Prepare the response
            self.response = pd.DataFrame([[self.model.name, utils.dict_to_sse_arg(self.model.best_params)]])
        except AttributeError:
            err = "Best parameters are not available as a parameter grid was not provided for cross validation."
            raise Exception(err)
        
        # Send the reponse table description to Qlik
        self._send_table_description("best_params")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def get_metrics(self):
        """
        Return metrics previously calculated during fit
        """
        
        # Get the model from cache or disk based on the model_name in request
        self._get_model_by_name()
        
        # Prepare the response data frame
        self.response = self.model.metrics_df
        
        # Send the reponse table description to Qlik
        if self.model.validation == "hold-out":
            if self.model.estimator_type == "classifier":
                self._send_table_description("metrics_clf")
            elif self.model.estimator_type == "regressor":
                self._send_table_description("metrics_reg")
        elif self.model.validation in ["k-fold", "timeseries"]:
            if self.model.estimator_type == "classifier":
                self._send_table_description("metrics_clf_cv")
            elif self.model.estimator_type == "regressor":
                self._send_table_description("metrics_reg_cv")
        else:
            err = "Metrics are not available. Make sure the machine learning pipeline includes K-fold cross validation or hold-out testing."
            raise Exception(err)
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response
    
    def get_keras_history(self):
        """
        Return history previously saved while fitting a Keras model.
        The history is avaialble as a DataFrame from the estimator.
        It provides metrics such as loss for each epoch during each run of the fit method.
        Columns will be ['iteration', 'epoch', 'loss'] and any other metrics calculated during training.
        """
        
        # Get the model from cache or disk based on the model_name in request
        self._get_model_by_name()

        assert self.model.using_keras, "Loss history is only available for Keras models"

        # Prepare the response using the histories data frame from the Keras model
        self.response = self.model.pipe.named_steps['estimator'].histories.copy()
        
        # Add the model name to the response
        self.response.insert(0, 'model_name', self.model.name)
        
        self._send_table_description("keras_history")
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(4)
        
        # Finally send the response
        return self.response

    def _set_params(self, estimator_args, scaler_args, execution_args, metric_args=None, dim_reduction_args=None):
        """
        Set input parameters based on the request.
        :
        :Refer to the sklearn API Reference for parameters avaialble for specific algorithms and scalers
        :http://scikit-learn.org/stable/modules/classes.html#api-reference
        :
        :Additional parameters used by this SSE are: 
        :overwrite, test_size, randon_state, compress, retain_data, debug
        :For details refer to the GitHub project: https://github.com/nabeel-oz/qlik-py-tools
        """
        
        # Set default values which will be used if execution arguments are not passed
        
        # Default parameters:
        self.model.overwrite = True
        self.model.debug = False
        self.model.test_size = 0.33
        self.model.cv = 0
        self.model.time_series_split = 0
        self.model.max_train_size = None
        self.model.random_state = 42
        self.model.compress = 3
        self.model.retain_data = False
        self.model.scale_hashed = True
        self.model.scale_vectors = True
        self.model.scaler = "StandardScaler"
        self.model.scaler_kwargs = {}
        self.model.estimator_kwargs = {}
        self.model.missing = "zeros"
        self.model.calc_feature_importances = False
        self.model.lags= None
        self.model.lag_target = False
        self.model.scale_target = False
        self.model.scale_lag_target= True
        self.model.make_stationary = None
        self.model.stationarity_lags = [1]
        self.model.using_keras = False
        self.model.current_sample_as_input = True
        self.model.prediction_periods = 1
        
        # Default metric parameters:
        if metric_args is None:
            self.model.metric_args = {}
        
        # Set execution parameters
                
        # If the execution key word arguments were included in the request, get the parameters and values
        if len(execution_args) > 0:
            
            # Transform the string of arguments into a dictionary
            execution_args = utils.get_kwargs(execution_args)
            
            # Set the overwite parameter if any existing model with the specified name should be overwritten
            if 'overwrite' in execution_args:
                self.model.overwrite = 'true' == execution_args['overwrite'].lower()
            
            # Set the test_size parameter that will be used to split the samples into training and testing data sets
            # Default value is 0.33, i.e. we use 66% of the samples for training and 33% for testing
            if 'test_size' in execution_args:
                self.model.test_size = utils.atof(execution_args['test_size'])

            # Enable K-fold cross validation. For more information see: http://scikit-learn.org/stable/modules/cross_validation.html#multimetric-cross-validation
            # Default value is 0 in which case a simple holdout strategy based on the test_size parameter is used.
            # If cv > 0 then the model is validated used K = cv folds and the test_size parameter is ignored.
            if 'cv' in execution_args:
                self.model.cv = utils.atoi(execution_args['cv'])
            
            # Enable timeseries backtesting using TimeSeriesSplit. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
            # This will select the a validation strategy appropriate for time series and sequential data.
            # The feature definitions must include an 'identifier' field which can be used to sort the series into the correct order.
            # The integer supplied in this parameter will split the data into the given number of subsets for training and testing.
            if 'time_series_split' in execution_args:
                self.model.time_series_split = utils.atoi(execution_args['time_series_split'])

            # This parameter can be used together with time_series_split.
            # It specifies the maximum samples to be used for training in each split, which allows for rolling/ walk forward validation.
            if 'max_train_size' in execution_args:
                self.model.max_train_size = utils.atoi(execution_args['max_train_size'])

            # Add lag observations to the feature matrix. Only applicable for Keras models.
            # An identifier field must be included in the feature definitions to correctly sort the data for this capability.
            # For e.g. if lags=2, features from the previous two samples will be concatenated as input features for the current sample.
            # This is useful for framing timeseries and sequence prediction problems into 3D or 4D data required for deep learning.
            if 'lags' in execution_args:
                self.model.lags = utils.atoi(execution_args['lags'])

            # Include targets in the lag observations
            # If True an additional feature will be created for each sample using the previous value of y 
            if 'lag_target' in execution_args:
                self.model.lag_target = 'true' == execution_args['lag_target'].lower()
            
            # Scale the target before fitting
            # The scaling will be inversed before predictions so they are returned in the original scale 
            if 'scale_target' in execution_args:
                self.model.scale_target = 'true' == execution_args['scale_target'].lower()

            # Scale lag values of the targets before fitting
            # Even if scale_target is set to false, the lag values of targets being used as features can be scaled by setting this to true 
            if 'scale_lag_target' in execution_args:
                self.model.scale_lag_target = 'true' == execution_args['scale_lag_target'].lower()

            # Make the target series more stationary. This only applies to sequence prediction problems.
            # Valid values are 'log' in which case we apply a logarithm to the target values,
            # or 'difference' in which case we transform the targets into variance from the previous value.
            # The transformation will be reversed before returning predictions.
            if 'make_stationary' in execution_args:
                self.model.make_stationary = execution_args['make_stationary'].lower()

                # Provide lags periods for differencing
                # By default the difference will be done with lag = 1. Alternate lags can be provided by passing a list of lags as a list.
                # e.g. 'stationarity_lags=1;12|list|int'
                if 'stationarity_lags' in execution_args:
                    self.model.stationarity_lags = utils.get_kwargs_by_type({'stationarity_lags': execution_args['stationarity_lags']})['stationarity_lags']

            # Specify if the current sample should be used as input to the model
            # This is to allow for models that only use lag observations to make future predictions
            if 'current_sample_as_input' in execution_args:
                self.model.current_sample_as_input = 'true' == execution_args['current_sample_as_input'].lower()

            # Specify the number of predictions expected from the model
            # This can be used to get a model to predict the next m periods given inputs for the previous n periods.
            # This is only valid for Keras models which have a final output layer with more than one node
            if 'prediction_periods' in execution_args:
                self.model.prediction_periods = utils.atoi(execution_args['prediction_periods'])
            
            # Seed used by the random number generator when generating the training testing split
            if 'random_state' in execution_args:
                self.model.random_state = utils.atoi(execution_args['random_state'])
            
            # Compression level between 1-9 used by joblib when saving the model
            if 'compress' in execution_args:
                self.model.compress = utils.atoi(execution_args['compress'])
                
            # Flag to determine if the training and test data should be saved in the model
            if 'retain_data' in execution_args:
                self.model.retain_data = 'true' == execution_args['retain_data'].lower()

            # Flag to determine if feature importances should be calculated when the fit method is called
            if 'calculate_importances' in execution_args:
                self.model.calc_feature_importances = 'true' == execution_args['calculate_importances'].lower()
                       
            # Set the debug option for generating execution logs
            # Valid values are: true, false
            if 'debug' in execution_args:
                self.model.debug = 'true' == execution_args['debug'].lower()
                
                # Additional information is printed to the terminal and logs if the paramater debug = true
                if self.model.debug:
                    # Increment log counter for the class. Each instance of the class generates a new log.
                    self.__class__.log_no += 1

                    # Create a log file for the instance
                    # Logs will be stored in ..\logs\SKLearn Log <n>.txt
                    self.logfile = os.path.join(os.getcwd(), 'logs', 'SKLearn Log {}.txt'.format(self.log_no))
                    
                    # Create dictionary of parameters to display for debug
                    self.exec_params = {"overwrite":self.model.overwrite, "test_size":self.model.test_size, "cv":self.model.cv,\
                    "time_series_split": self.model.time_series_split, "max_train_size":self.model.max_train_size, "lags":self.model.lags,\
                    "lag_target":self.model.lag_target, "scale_target":self.model.scale_target, "make_stationary":self.model.make_stationary,\
                    "random_state":self.model.random_state, "compress":self.model.compress, "retain_data":self.model.retain_data,\
                    "calculate_importances": self.model.calc_feature_importances, "debug":self.model.debug}

                    self._print_log(1)
        
        # If the scaler key word arguments were included in the request, get the parameters and values
        if len(scaler_args) > 0:
            
            # Transform the string of arguments into a dictionary
            scaler_args = utils.get_kwargs(scaler_args)
                   
            # Set scaler arguments that will be used when preprocessing the data
            # Valid values are: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler and QuantileTransformer
            # More information here: http://scikit-learn.org/stable/modules/preprocessing.html
            if 'scaler' in scaler_args:
                self.model.scaler = scaler_args.pop('scaler')
                
                if 'missing' in scaler_args:
                    self.model.missing = scaler_args.pop('missing').lower()
                
                if 'scale_hashed' in scaler_args:
                    self.model.scale_hashed = 'true' == scaler_args.pop('scale_hashed').lower()
                
                if 'scale_vectors' in scaler_args:
                    self.model.scale_vectors = 'true' == scaler_args.pop('scale_vectors').lower()
                
                # Get the rest of the scaler parameters, converting values to the correct data type
                self.model.scaler_kwargs = utils.get_kwargs_by_type(scaler_args) 
            else:
                err = "Arguments for scaling did not include the scaler name e.g StandardScaler"
                raise Exception(err)
            
        # If the estimator key word arguments were included in the request, get the parameters and values
        if len(estimator_args) > 0:
            
            # Transform the string of arguments into a dictionary
            estimator_args = utils.get_kwargs(estimator_args)
                   
            # Set estimator arguments that will be used when preprocessing the data
            # The parameters available will depend on the selected estimator
            # More information here: http://scikit-learn.org/stable/modules/classes.html#api-reference
            if 'estimator' in estimator_args:
                self.model.estimator = estimator_args.pop('estimator')
                
                # Set the estimator type for the model
                if self.model.estimator in self.classifiers:
                    self.model.estimator_type = "classifier"
                elif self.model.estimator in self.regressors:
                    self.model.estimator_type = "regressor"
                elif self.model.estimator in self.decomposers:
                    self.model.estimator_type = "decomposer"
                elif self.model.estimator in self.clusterers:
                    self.model.estimator_type = "clusterer"
                else:
                    err = "Unknown estimator class: {0}".format(self.model.estimator)
                    raise Exception(err)

                # Get the rest of the estimator parameters, converting values to the correct data type
                self.model.estimator_kwargs = utils.get_kwargs_by_type(estimator_args)  
            else:
                err = "Arguments for estimator did not include the estimator class e.g. RandomForestClassifier"
                raise Exception(err)
        
        # If key word arguments for model evaluation metrics are included in the request, get the parameters and values
        if metric_args is not None and len(metric_args) > 0:
            # Transform the string of arguments into a dictionary
            metric_args = utils.get_kwargs(metric_args)
            
            # Get the metric parameters, converting values to the correct data type
            self.model.metric_args = utils.get_kwargs_by_type(metric_args)  
        
        # If key word arguments for dimensionality reduction are included in the request, get the parameters and values
        if dim_reduction_args is not None and len(dim_reduction_args) > 0:
            # Transform the string of arguments into a dictionary
            dim_reduction_args = utils.get_kwargs(dim_reduction_args)
                   
            # Set dim_reduction arguments that will be used after preprocessing the data
            # The parameters available will depend on the selected dimensionality reduction method
            # Acceptable classes are PCA, KernelPCA, IncrementalPCA, TruncatedSVD
            # More information here: http://scikit-learn.org/stable/modules/classes.html#api-reference
            if 'reduction' in dim_reduction_args:
                self.model.reduction = dim_reduction_args.pop('reduction')
                
                # Get the rest of the dim_reduction parameters, converting values to the correct data type
                self.model.dim_reduction_args = utils.get_kwargs_by_type(dim_reduction_args)  
            else:
                err = "Arguments for dimensionality reduction did not include the class e.g. PCA"
                raise Exception(err)
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(2)
    
    def _set_grid_params(self, param_grid, grid_search_args):
        """
        Set up the grid search parameters to be used later for sklearn.model_selection.GridSearchCV
        param_grid is a series with the hyperparameters. The series can have multiple rows.
        grid_search_args is a string with the arguments for the call to GridSearchCV.
        """
        
        # If key word arguments for the grid search are included in the request, get the parameters and values
        if len(grid_search_args) > 0:
            # Transform the string of arguments into a dictionary
            grid_search_args = utils.get_kwargs(grid_search_args)
            
            # Get the metric parameters, converting values to the correct data type
            self.model.grid_search_args = utils.get_kwargs_by_type(grid_search_args)

            # The refit parameter must be True, so this is ignored if passed in the arguments
            self.model.grid_search_args["refit"] = True
        else:
            self.model.grid_search_args = {}
        
        # If key word arguments for the grid search are included in the request, get the parameters and values
        if len(param_grid) > 0:
            # Transform the parameter grid dataframe into a list of dictionaries
            self.model.param_grid = list(param_grid.apply(utils.get_kwargs).apply(utils.get_kwargs_by_type))
        else:
            err = "An empty string is not a valid input for the param_grid argument"
            raise Exception(err)
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(9)

    def _get_model_by_name(self):
        """
        Get a previously saved model using the model_name
        """
        
        # Interpret the request data based on the expected row and column structure
        row_template = ['strData']
        col_headers = ['model_name']
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)
        
        # Initialize the persistent model
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.loc[0, 'model_name']
        
        # Get the model from cache or disk
        self._get_model()
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(3)
    
    def _get_model_and_data(self, target=False, set_feature_def=False, ordered_data=False):
        """
        Get samples and targets based on the request and an existing model's feature definitions.
        If target=False, just return the samples.
        If set_feature_def=True, set the feature definitions for the model.
        If ordered_data=True, the request is expected to have a key field which will be used as an index. 
        The index will be stored in its original and sorted form in instance variables self.original_index and self.sorted_index
        """
    
        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'strData']
        col_headers = ['model_name', 'n_features']
        features_col_num = 1

        # If True, the request is expected to have a key field which will be used as an index.
        if ordered_data:
            row_template.append('strData')
            col_headers.insert(1, 'key')
            features_col_num = 2
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)

        if ordered_data:
            # Set the key column as the index
            self.request_df.set_index("key", drop=False, inplace=True)
            # Store this index so we can reset the data to its original sort order
            self.original_index = self.request_df.index.copy()
        
        # Initialize the persistent model
        self.model = PersistentModel()
        
        # Get the model name from the request dataframe
        self.model.name = self.request_df.iloc[0, 0]
        
        # Get the model from cache or disk
        self._get_model()
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(3)

        # If the model requires lag targets, the data is always expected to contain targets
        if not target:
            target = self.model.lag_target
        
        # Get the expected features for this request
        features_df = self.model.original_features_df.copy()
        
        # Set features that are not expected in the current request
        if not set_feature_def:
            exclude = ["excluded"]

            if not target:
                exclude.append("target")
            if not ordered_data:
                exclude.append("identifier")

            # Exclude columns that are not expected in the request data
            exclusions = features_df['variable_type'].isin(exclude)
            features_df = features_df.loc[~exclusions]
        
        try:
            # Split the features provided as a string into individual columns
            samples_df = pd.DataFrame([x[features_col_num].split("|") for x in self.request_df.values.tolist()],\
                                            columns=features_df.loc[:,"name"].tolist(),\
                                            index=self.request_df.index)
        except AssertionError as ae:
            err = "The number of input columns do not match feature definitions. Ensure you are using the | delimiter and providing the correct features."
            err += "\n\nSample rows:\n{}\n\nExpected Features:\n{}\n".format(self.request_df.head(3), features_df.loc[:,"name"].tolist())
            raise AssertionError(err) from ae
        
        # Convert the data types based on feature definitions and sort by the unique identifier (if defined in the definitions)
        samples_df = utils.convert_types(samples_df, features_df, sort=True)

        if ordered_data:
            # Store the sorted index 
            self.sorted_index = samples_df.index.copy()
        
        if target:
            # Get the target feature
            target_name = self.model.original_features_df.loc[self.model.original_features_df["variable_type"] == "target"].index[0]

            # Get the target data
            target_df = samples_df.loc[:,[target_name]]
        
        # Update the feature definitions dataframe
        if set_feature_def:
            # Get the features to be excluded from the model
            exclusions = features_df['variable_type'].isin(["excluded", "target", "identifier"])
            # Store the featuer definitions except exclusions
            features_df = features_df.loc[~exclusions]
            self.model.features_df = features_df
        
        # Remove excluded features, target and identifier from the data
        samples_df = samples_df[features_df.index.tolist()]
        
        if target:
            return [samples_df, target_df]
        else:
            return samples_df
    
    def _add_lags(self, X, y=None, extrapolate=1, update_features_df=False):
        """
        Add lag observations to X.
        If y is available and self.model.lag_target is True, the previous target will become an additional feature in X.
        Feature definitions for the model will be updated accordingly.
        """

        # Add lag target to the features if required
        # This will create an additional feature for each sample i.e. the previous value of y 
        if y is not None and self.model.lag_target:
            X["previous_y"] = y.shift(1)
            
            if update_features_df:
                # Check the target's data type
                dt = 'float' if is_numeric_dtype(y.iloc[:,0]) else 'str'
                # Set the preprocessing feature strategy for the lag targets
                if self.model.estimator_type == 'classifier':
                    fs = 'one hot encoding' 
                elif self.model.scale_lag_target and not self.model.scale_target:
                    fs = 'scaling'
                else:
                    fs = 'none'
                self.model.scale_lag_target
                # Update feature definitions for the model
                self.model.features_df.loc['previous_y'] = [self.model.name, 'previous_y', 'feature', dt, fs, '']

        if self.model.lags:
            # Add the lag observations
            X = utils.add_lags(X, lag=self.model.lags, extrapolate=extrapolate, dropna=True, suffix="t")
            
            if update_features_df:
                # Duplicate the feature definitions by the number of lags
                self.model.features_df = pd.concat([self.model.features_df] * (self.model.lags+extrapolate))
                # Set the new feature names as the index of the feature definitions data frame
                self.model.features_df['name'] = X.columns
                self.model.features_df = self.model.features_df.set_index('name', drop=True)

        if self.model.debug:
            self._print_log(11, data=X)

        return X
    
    def _keras_update_shape(self, prep):
        """
        Update the input shape for the Keras architecture.
        We do this by running preprocessing on the training data frame to get the final number of features.
        """

        # Run preprocessing on the training data
        X_transform = prep.fit_transform(self.X_train)

        # If the input shape has not been specified, it is simply the number of features in X_transform
        if 'input_shape' not in self.model.first_layer_kwargs:
            self.model.first_layer_kwargs['input_shape'] = tuple([X_transform.shape[1]])
        # Else update the input shape based on the number of features after preprocessing
        else:
            # Transform to a list to make the input_shape mutable
            self.model.first_layer_kwargs['input_shape'] = list(self.model.first_layer_kwargs['input_shape'])
            # Update the number of features based on X_transform
            if self.model.lags:
                self.model.first_layer_kwargs['input_shape'][-1] = X_transform.shape[1]//(self.model.lags + (1 if self.model.current_sample_as_input else 0))
            else:
                self.model.first_layer_kwargs['input_shape'][-1] = X_transform.shape[1]//np.prod(self.model.first_layer_kwargs['input_shape'][:-1])
            # Transform back to a tuple as required by Keras
            self.model.first_layer_kwargs['input_shape'] = tuple(self.model.first_layer_kwargs['input_shape'])
        
        # Ensure the Architecture has been updated
        self.model.architecture.iloc[0, 2]['input_shape'] = self.model.first_layer_kwargs['input_shape']
        
        # 2D, 3D and 4D data is valid. 
        # e.g. The input_shape can be a tuple of (subsequences, timesteps, features), with subsequences and timesteps as optional.
        # A 4D shape may be valid for e.g. a ConvLSTM with (timesteps, rows, columns, features) 
        if len(self.model.first_layer_kwargs['input_shape']) > 5:
            err = "Unsupported input_shape: {}".format(self.model.first_layer_kwargs['input_shape'])
            raise Exception(err)

    @staticmethod
    def _keras_build_fn(architecture=None, prediction_periods=1):
        """
        Create and compile a Keras Sequential model based on the model's architecture dataframe.
        
        The architecture dataframe should define the layers and compilation parameters for a Keras sequential model.
        Each layer has to be defined across three columns: layer, args, kwargs.
        The first layer should define the input_shape. This SSE does not support the alternative input_dim argument.
        The final row of the dataframe should define the compilation parameters.

        For example:
          layer        args    kwargs
        0 Dense        [12]    {'input_shape': (8,), activation': 'relu'}
        1 Dropout      [0.25]  {}
        2 Dense        [8]     {'activation': 'relu'}
        3 Dense        [1]     {'activation': 'sigmoid'}
        4 Compilation  []      {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']}

        If you want to specify parameters for the optimizer, you can add that as the second last row of the architecture:
        ...
        4 SGD          []      {'lr': 0.01, 'clipvalue': 0.5}
        5 Compilation  []      {'loss': 'binary_crossentropy'}

        Layer wrappers can be used by including them in the layer name followed by a space:
        0 TimeDistributed Dense ...
        ...
        For the Bidirectional wrapper the merge_mode parameter can be specified in the kwargs.

        When using recurrent or convolutional layers you will need to pass a valid input_shape to reshape the data.
        Additionally, the feature definitions should include an 'identifier' variable that will be used for reshaping.
        E.g. The identifier is a date field and you want to use a LSTM with 10 time steps on a dataset with 20 features.
        0 LSTM         [64]    {'input_shape': (10,20), 'activation': 'relu''
        
        For further information on the columns refer to the project documentation: 
        https://github.com/nabeel-oz/qlik-py-tools
        """
        
        # List of optimizers that can be specified in the architecture
        optimizers = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD']
        
        # The model definition should contain at least one layer and the compilation parameters
        if architecture is None or len(architecture) < 2:
            err = "Invalid Keras architecture. Expected at least one layer and compilation parameters."
            raise Exception(err)
        # The last row of the model definition should contain compilation parameters
        elif not architecture.iloc[-1,0].capitalize() in ['Compile', 'Compilation']:
            err = "Invalid Keras architecture. The last row of the model definition should provide 'Compile' parameters."
            raise Exception(err)
        
        # sys.stdout.write("Architecture Data Frame in _keras_build_fn:\n{}\n\n".format(architecture.to_string()))

        neural_net = keras.models.Sequential()

        for i in architecture.index:
            # Name items in the row for easy access
            name, args, kwargs = architecture.iloc[i,0], architecture.iloc[i,1], architecture.iloc[i,2]

            # The last row of the DataFrame should provide compilation keyword arguments
            if i == max(architecture.index):
                # Check if an optimizer with custom parameters has been defined
                try:
                    kwargs = kwargs.copy() # Copy so that we don't modify the architecture dataframe
                    kwargs['optimizer'] = opt
                except UnboundLocalError:
                    pass
                
                # Compile the model
                neural_net.compile(**kwargs)
            # Watch out for a row providing optimizer parameters
            elif name in optimizers:
                opt = getattr(keras.optimizers, name)(**kwargs) 
            # All other rows of the DataFrame define the model architecture
            else:
                # Check if the name includes a layer wrapper e.g. TimeDistributed Dense
                names = name.split(' ')
                if len(names) == 2:
                    wrapper = names[0]
                    name = names[1]
                    
                    # Get wrapper kwargs
                    wrapper_kwargs = dict()
                    if 'merge_mode' in kwargs:
                        wrapper_kwargs['merge_mode'] = kwargs.pop('merge_mode')
                else:
                    wrapper = None

                # Create a keras layer of the required type with the provided positional and keyword arguments
                layer = getattr(keras.layers, name)(*args, **kwargs)

                if wrapper:
                    # Create the layer wrapper
                    wrapper = getattr(keras.layers, wrapper)(layer, **wrapper_kwargs)
                    # Add the layer wrapper to the model
                    neural_net.add(wrapper)    
                else:
                    # Add the layer to the model
                    neural_net.add(layer)
        
        # Get the number of nodes for the final layer
        output_features = neural_net.layers[-1].get_config()['units']
        assert prediction_periods == output_features, "The number of nodes in the final layer of the network must match the prediction_periods execution argument. Expected {} nodes but got {}.".format(prediction_periods, output_features)
        
        return neural_net

    def _cross_validate(self, fit_params={}):
        """
        Perform K-fold cross validation on the model for the dataset provided in the request.
        """

        # Flatten the true labels for the training data
        y_train = self.y_train.values if self.y_train.shape[1] > 1 else self.y_train.values.ravel()

        if self.model.estimator_type == "classifier":

            # Get unique labels for classification
            labels = np.unique(y_train)

            # Set up a dictionary for the scoring metrics
            scoring = {'accuracy':'accuracy'}

            # Prepare arguments for the scorers
            metric_args = self.model.metric_args
            
            if 'average' in metric_args and metric_args['average'] is not None:
                # If the score is being averaged over classes a single scorer per metric is sufficient
                scoring['precision'] = metrics.make_scorer(metrics.precision_score, **metric_args)
                scoring['recall'] = metrics.make_scorer(metrics.recall_score, **metric_args)
                scoring['fscore'] = metrics.make_scorer(metrics.f1_score, **metric_args)

                output_format = "clf_overall"
            else:
                # If there is no averaging we will need multiple scorers; one for each class
                for label in labels:
                    metric_args['pos_label'] = label
                    metric_args['labels'] = [label]
                    scoring['precision_'+str(label)] = metrics.make_scorer(metrics.precision_score, **metric_args)
                    scoring['recall_'+str(label)] = metrics.make_scorer(metrics.recall_score, **metric_args)
                    scoring['fscore_'+str(label)] = metrics.make_scorer(metrics.f1_score, **metric_args)
                
                output_format = "clf_classes"

        elif self.model.estimator_type == "regressor":
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'explained_variance']
        
        # Perform cross validation using the training data and the model pipeline
        scores = cross_validate(self.model.pipe, self.X_train, y_train, scoring=scoring, cv=self.model.cv, fit_params=fit_params, return_train_score=False)

        # Prepare the metrics data frame according to the output format
        if self.model.estimator_type == "classifier":           
            # Get cross validation predictions for the confusion matrix
            y_pred = cross_val_predict(self.model.pipe, self.X_train, y_train, cv=self.model.cv, fit_params=fit_params)

            # Prepare the confusion matrix and add it to the model
            self._prep_confusion_matrix(y_train, y_pred, labels)

            # Create an empty data frame to set the structure
            metrics_df = pd.DataFrame(columns=["class", "accuracy", "accuracy_std", "precision", "precision_std", "recall",\
            "recall_std", "fscore", "fscore_std"])

            if output_format == "clf_overall":               
                # Add the overall metrics to the data frame
                metrics_df.loc[0] = ["overall", np.average(scores["test_accuracy"]), np.std(scores["test_accuracy"]),\
                np.average(scores["test_precision"]), np.std(scores["test_precision"]),\
                np.average(scores["test_recall"]), np.std(scores["test_recall"]),\
                np.average(scores["test_fscore"]), np.std(scores["test_fscore"])]

            elif output_format == "clf_classes":
                # Add accuracy which is calculated at an overall level
                metrics_df.loc[0] = ["overall", np.average(scores["test_accuracy"]), np.std(scores["test_accuracy"]),\
                np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

                # Add the metrics for each class to the data frame
                for i, label in enumerate(labels):
                    metrics_df.loc[i+1] = [label, np.NaN, np.NaN, np.average(scores["test_precision_"+str(label)]),\
                    np.std(scores["test_precision_"+str(label)]), np.average(scores["test_recall_"+str(label)]),\
                    np.std(scores["test_recall_"+str(label)]), np.average(scores["test_fscore_"+str(label)]),\
                    np.std(scores["test_fscore_"+str(label)])]
            
            # Finalize the structure of the result DataFrame
            metrics_df.loc[:,"model_name"] = self.model.name
            metrics_df = metrics_df.loc[:,["model_name", "class", "accuracy", "accuracy_std", "precision", "precision_std", "recall",\
            "recall_std", "fscore", "fscore_std"]]

            # Add the score to the model
            self.model.score =  metrics_df["accuracy"].values[0]

        elif self.model.estimator_type == "regressor":
            # Create an empty data frame to set the structure
            metrics_df = pd.DataFrame(columns=["r2_score", "r2_score_std", "mean_squared_error", "mean_squared_error_std",\
            "mean_absolute_error", "mean_absolute_error_std", "median_absolute_error", "median_absolute_error_std",\
            "explained_variance_score", "explained_variance_score_std"])
            
            # Add the overall metrics to the data frame
            metrics_df.loc[0] = [np.average(scores["test_r2"]), np.std(scores["test_r2"]),\
            np.average(scores["test_neg_mean_squared_error"]), np.std(scores["test_neg_mean_squared_error"]),\
            np.average(scores["test_neg_mean_absolute_error"]), np.std(scores["test_neg_mean_absolute_error"]),\
            np.average(scores["test_neg_median_absolute_error"]), np.std(scores["test_neg_median_absolute_error"]),\
            np.average(scores["test_explained_variance"]), np.std(scores["test_explained_variance"])]
        
            # Finalize the structure of the result DataFrame
            metrics_df.loc[:,"model_name"] = self.model.name
            metrics_df = metrics_df.loc[:,["model_name", "r2_score", "r2_score_std", "mean_squared_error", "mean_squared_error_std",\
            "mean_absolute_error", "mean_absolute_error_std", "median_absolute_error", "median_absolute_error_std",\
            "explained_variance_score", "explained_variance_score_std"]]

            # Add the score to the model
            self.model.score =  metrics_df["r2_score"].values[0]

        # Save the metrics_df to the model
        self.model.metrics_df = metrics_df
    
    def _prep_confusion_matrix(self, y_test, y_pred, labels):
        """
        Calculate a confusion matrix and add it to the model as a data frame suitable for Qlik
        """

        # Calculate confusion matrix and flatten it to a simple array
        if len(y_test.shape) == 1:
            confusion_array = metrics.confusion_matrix(y_test, y_pred).ravel()

            # Structure into a DataFrame suitable for Qlik
            result = []
            i = 0
            for t in labels:
                for p in labels:
                    result.append([str(t), str(p), confusion_array[i]])
                    i = i + 1
            self.model.confusion_matrix = pd.DataFrame(result, columns=["true_label", "pred_label", "count"])
            self.model.confusion_matrix.insert(0, "model_name", self.model.name)
        # Handle confusion matrix format for multi-label classification
        else:
            confusion_array = metrics.multilabel_confusion_matrix(y_test, y_pred)
            result = pd.DataFrame(confusion_array.reshape(-1, 4), columns=["true_negative", "false_positive", "false_negative", "true_positive"])
            self.model.confusion_matrix = pd.DataFrame(np.arange(len(confusion_array)), columns=["step"])
            self.model.confusion_matrix = pd.concat([self.model.confusion_matrix, result], axis=1)
            self.model.confusion_matrix.insert(0, "model_name", self.model.name)
                
    def _calc_importances(self, X=None, y=None):
        """
        Calculate feature importances.
        Importances are calculated using the Skater library to provide this capability for all sklearn algorithms.
        For more information: https://www.datascience.com/resources/tools/skater
        """
        
        # Fill null values in the test set according to the model settings
        X_test = utils.fillna(X, method=self.model.missing)
        
        # Calculate model agnostic feature importances using the skater library
        interpreter = Interpretation(X_test, training_labels=y, feature_names=self.model.features_df.index.tolist())
        
        if self.model.estimator_type == "classifier":
            try:
                # We use the predicted probabilities from the estimator if available
                predictor = self.model.pipe.predict_proba

                # Set up keyword arguments accordingly
                imm_kwargs = {"probability": True}
            except AttributeError:
                # Otherwise we simply use the predict method
                predictor = self.model.pipe.predict

                # Set up keyword arguments accordingly
                imm_kwargs = {"probability": False, "unique_values": self.model.pipe.classes_}
            
            # Set up a skater InMemoryModel to calculate feature importances
            imm = InMemoryModel(predictor, examples = X_test[:10], model_type="classifier", **imm_kwargs)
        
        elif self.model.estimator_type == "regressor":
            # Set up a skater InMemoryModel to calculate feature importances using the predict method
            imm = InMemoryModel(self.model.pipe.predict, examples = X_test[:10], model_type="regressor")
        
        # Add the feature importances to the model as a sorted data frame
        self.model.importances = interpreter.feature_importance.feature_importance(imm, progressbar=False, ascending=False)
        self.model.importances = pd.DataFrame(self.model.importances).reset_index()
        self.model.importances.columns = ["feature_name", "importance"]

    def _send_table_description(self, variant):
        """
        Send the table description to Qlik as meta data.
        Only used when the SSE is called from the Qlik load script.
        """
        
        # Set up the table description to send as metadata to Qlik
        self.table = SSE.TableDescription()
        self.table.name = "SSE-Response"
        self.table.numberOfRows = len(self.response)

        # Set up fields for the table
        if variant == "setup":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="result")
            self.table.fields.add(name="timestamp")
        elif variant == "features":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="sort_order", dataType=1)
            self.table.fields.add(name="feature")
            self.table.fields.add(name="var_type")
            self.table.fields.add(name="data_type")
            self.table.fields.add(name="strategy")
            self.table.fields.add(name="strategy_args")
        elif variant == "fit":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="result")
            self.table.fields.add(name="time_stamp")
            self.table.fields.add(name="score_result")
            self.table.fields.add(name="score", dataType=1)
        elif variant == "metrics_clf":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="class")
            self.table.fields.add(name="accuracy", dataType=1)
            self.table.fields.add(name="precision", dataType=1)
            self.table.fields.add(name="recall", dataType=1)
            self.table.fields.add(name="fscore", dataType=1)
            self.table.fields.add(name="support", dataType=1)
        elif variant == "metrics_clf_cv":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="class")
            self.table.fields.add(name="accuracy", dataType=1)
            self.table.fields.add(name="accuracy_std", dataType=1)
            self.table.fields.add(name="precision", dataType=1)
            self.table.fields.add(name="precision_std", dataType=1)
            self.table.fields.add(name="recall", dataType=1)
            self.table.fields.add(name="recall_std", dataType=1)
            self.table.fields.add(name="fscore", dataType=1)
            self.table.fields.add(name="fscore_std", dataType=1)
        elif variant == "metrics_reg":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="r2_score", dataType=1)
            self.table.fields.add(name="mean_squared_error", dataType=1)
            self.table.fields.add(name="mean_absolute_error", dataType=1)
            self.table.fields.add(name="median_absolute_error", dataType=1)
            self.table.fields.add(name="explained_variance_score", dataType=1)
        elif variant == "metrics_reg_cv":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="r2_score", dataType=1)
            self.table.fields.add(name="r2_score_std", dataType=1)
            self.table.fields.add(name="mean_squared_error", dataType=1)
            self.table.fields.add(name="mean_squared_error_std", dataType=1)
            self.table.fields.add(name="mean_absolute_error", dataType=1)
            self.table.fields.add(name="mean_absolute_error_std", dataType=1)
            self.table.fields.add(name="median_absolute_error", dataType=1)
            self.table.fields.add(name="median_absolute_error_std", dataType=1)
            self.table.fields.add(name="explained_variance_score", dataType=1)
            self.table.fields.add(name="explained_variance_score_std", dataType=1)
        elif variant == "confusion_matrix":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="true_label")
            self.table.fields.add(name="pred_label")
            self.table.fields.add(name="count", dataType=1)
        elif variant == "confusion_matrix_multi":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="step", dataType=1)
            self.table.fields.add(name="true_negative", dataType=1)
            self.table.fields.add(name="false_positive", dataType=1)
            self.table.fields.add(name="false_negative", dataType=1)
            self.table.fields.add(name="true_positive", dataType=1)
        elif variant == "importances":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="feature_name")
            self.table.fields.add(name="importance", dataType=1)
        elif variant == "predict":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="key")
            self.table.fields.add(name="prediction")
        elif variant == "expression":
            self.table.fields.add(name="result")
        elif variant == "best_params":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="best_params")
        elif variant == "cluster":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="key")
            self.table.fields.add(name="label")
        elif variant == "reduce":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="key")
            # Add a variable number of columns depending on the response
            for i in range(self.response.shape[1]-2):
                self.table.fields.add(name="dim_{0}".format(i+1), dataType=1)
        elif variant == "keras_history":
            self.table.fields.add(name="model_name")
            # Add columns from the Keras model's history
            for i in range(1, self.response.shape[1]):
                self.table.fields.add(name=self.response.columns[i], dataType=1)
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(5)
            
        # Send table description
        table_header = (('qlik-tabledescription-bin', self.table.SerializeToString()),)
        self.context.send_initial_metadata(table_header)
    
    def _get_model(self, use_cache=True):
        """
        Get the model from the class model cache or disk.
        Update the cache if loading from disk.
        Return the model.
        """
        
        if use_cache and self.model.name in self.__class__.model_cache:
            # Load the model from cache
            self.model = self.__class__.model_cache[self.model.name]

            # Refresh the keras model to avoid tensorflow errors 
            if self.model.using_keras and hasattr(self.model, 'pipe'):
                self._keras_refresh()
            
            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(6)
        else:
            # Load the model from disk
            self.model = self.model.load(self.model.name, self.path) 

            # Debug information is printed to the terminal and logs if the paramater debug = true
            if self.model.debug:
                self._print_log(7)
            
            # Update the cache to keep this model in memory
            self._update_cache()
    
    def _update_cache(self):
        """
        Maintain a cache of recently used models at the class level
        """
        
        # Check if the model cache is full
        if self.__class__.cache_limit == len(self.__class__.model_cache):
            # Remove the oldest item from the cache if exceeding cache limit
            self.__class__.model_cache.popitem(last=False)
        
        # Remove the obsolete version of the model from the cache
        if self.model.name in self.__class__.model_cache:
            del self.__class__.model_cache[self.model.name]
        
        # Add the current model to the cache
        self.__class__.model_cache[self.model.name] = self.model
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(8)
    
    def _keras_refresh(self):
        """
        Avoid tensorflow errors for keras models by clearing the session and reloading from disk
        https://github.com/tensorflow/tensorflow/issues/14356
        https://stackoverflow.com/questions/40785224/tensorflow-cannot-interpret-feed-dict-key-as-tensor
        """
        kerasbackend.clear_session()

        # Load the keras model architecture and weights from disk
        keras_model = keras.models.load_model(self.path + self.model.name + '.h5')
        keras_model._make_predict_function()
        # Point the model's estimator in the sklearn pipeline to the keras model architecture and weights 
        self.model.pipe.named_steps['estimator'].model = keras_model
    
    def _print_log(self, step, data=None):
        """
        Output useful information to stdout and the log file if debugging is required.
        :step: Print the corresponding step in the log
        """
        
        # Set mode to append to log file
        mode = 'a'

        if self.logfile is None:
            # Increment log counter for the class. Each instance of the class generates a new log.
            self.__class__.log_no += 1

            # Create a log file for the instance
            # Logs will be stored in ..\logs\SKLearn Log <n>.txt
            self.logfile = os.path.join(os.getcwd(), 'logs', 'SKLearn Log {}.txt'.format(self.log_no))
        
        if step == 1:
            # Output log header
            output = "\nSKLearnForQlik Log: {0} \n\n".format(time.ctime(time.time()))
            # Set mode to write new log file
            mode = 'w'
                
        elif step == 2:
            # Output the parameters
            output = "Model Name: {0}\n\n".format(self.model.name)
            output += "Execution arguments: {0}\n\n".format(self.exec_params)
            
            try:
                output += "Scaler: {0}, missing: {1}, scale_hashed: {2}, scale_vectors: {3}\n".format(\
                self.model.scaler, self.model.missing,self.model.scale_hashed, self.model.scale_vectors)
                output += "Scaler kwargs: {0}\n\n".format(self.model.scaler_kwargs)
            except AttributeError:
                output += "scale_hashed: {0}, scale_vectors: {1}\n".format(self.model.scale_hashed, self.model.scale_vectors)

            try:
                if self.model.dim_reduction:
                    output += "Reduction: {0}\nReduction kwargs: {1}\n\n".format(self.model.reduction, self.model.dim_reduction_args)
            except AttributeError:
                pass
            
            output += "Estimator: {0}\nEstimator kwargs: {1}\n\n".format(self.model.estimator, self.model.estimator_kwargs)
                
        elif step == 3:                    
            # Output the request dataframe
            output = "REQUEST: {0} rows x cols\nSample Data:\n\n".format(self.request_df.shape)
            output += "{0}\n...\n{1}\n\n".format(self.request_df.head().to_string(), self.request_df.tail().to_string())
        
        elif step == 4:
            # Output the response dataframe/series
            output = "RESPONSE: {0} rows x cols\nSample Data:\n\n".format(self.response.shape)
            output += "{0}\n...\n{1}\n\n".format(self.response.head().to_string(), self.response.tail().to_string())
                 
        elif step == 5:
            # Print the table description if the call was made from the load script
            output = "\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table)
        
        elif step == 6:
            # Message when model is loaded from cache
            output = "\nModel {0} loaded from cache.\n\n".format(self.model.name)
            
        elif step == 7:
            # Message when model is loaded from disk
            output = "\nModel {0} loaded from disk.\n\n".format(self.model.name)
            
        elif step == 8:
            # Message when cache is updated
            output = "\nCache updated. Models in cache:\n{0}\n\n".format([k for k,v in self.__class__.model_cache.items()])
        
        elif step == 9:
            # Output when a parameter grid is set up
            output = "Model Name: {0}, Estimator: {1}\n\nGrid Search Arguments: {2}\n\nParameter Grid: {3}\n\n".\
            format(self.model.name, self.model.estimator, self.model.grid_search_args, self.model.param_grid)
        
        elif step == 10:
            # self.model.estimator_kwargs['architecture']
            output = "\nKeras architecture added to Model {0}:\n\n{1}\n\n".format(self.model.name,\
            self.model.architecture.to_string())

        elif step == 11:
            # Output after adding lag observations to input data
            output = "Lag observations added ({0} per sample). New input shape of X is {1}.\n\n".format(self.model.lags, data.shape)
            output += "Feature Definitions:\n{0}\n\n".format(self.model.features_df.to_string())
            output += "Sample Data:\n{0}\n...\n{1}\n\n".format(data.head(5).to_string(), data.tail(5).to_string())
                        
        sys.stdout.write(output)
        with open(self.logfile, mode, encoding='utf-8') as f:
            f.write(output)

    def _print_exception(self, s, e):
        """
        Output exception message to stdout and also to the log file if debugging is required.
        :s: A description for the error
        :e: The exception
        """
        
        # Output exception message
        sys.stdout.write("\n{0}: {1} \n\n".format(s, e))
        
        if self.model.debug:
            with open(self.logfile,'a') as f:
                f.write("\n{0}: {1} \n\n".format(s, e))