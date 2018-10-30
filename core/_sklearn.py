import os
import sys
import ast
import time
import string
import locale
import pathlib
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict

# Turn off warnings by default
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

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

import _utils as utils
from _machine_learning import Preprocessor, PersistentModel
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
                            "OrthogonalMatchingPursuitCV":OrthogonalMatchingPursuitCV}
        
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
                            "DecisionTreeClassifier", "ExtraTreeClassifier"] 
        
        self.regressors = ["DummyRegressor", "AdaBoostRegressor", "BaggingRegressor", "ExtraTreesRegressor",\
                           "GradientBoostingRegressor", "RandomForestRegressor", "GaussianProcessRegressor",\
                           "LinearRegression", "PassiveAggressiveRegressor", "RANSACRegressor", "Ridge", "RidgeCV"\
                           "SGDRegressor", "TheilSenRegressor", "KNeighborsRegressor", "RadiusNeighborsRegressor",\
                           "MLPRegressor", "LinearSVR", "NuSVR", "SVR", "DecisionTreeRegressor",\
                           "ExtraTreeRegressor", "ARDRegression", "BayesianRidge", "ElasticNet", "ElasticNetCV",\
                           "HuberRegressor", "Lars", "LarsCV", "Lasso", "LassoCV", "LassoLars", "LassoLarsCV",\
                           "LassoLarsIC", "MultiTaskLasso", "MultiTaskElasticNet", "MultiTaskLassoCV", "MultiTaskElasticNetCV",\
                           "OrthogonalMatchingPursuit", "OrthogonalMatchingPursuitCV"]
    
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
        models = ", ".join([str(p).split("\\")[-1] for p in list(pathlib.Path(self.path).glob(search_pattern))])
        
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
        self.model = self.model.save(self.model.name, self.path, self.model.compress)
        
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
               
        # Persist the model to disk
        self.model = self.model.save(self.model.name, self.path, self.model.compress)
        
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
        self.model = self.model.save(self.model.name, self.path, self.model.compress)
        
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
    
    def fit(self):
        """
        Train and test the model based on the provided dataset
        """
        
        # Open an existing model and get the training & test dataset and targets
        train_test_df, target_df = self._get_model_and_data()
        
        # Check that the estimator is an supervised ML algorithm
        if self.model.estimator_type not in ["classifier", "regressor"]:
            err = "Incorrect usage. The estimator specified is not a known classifier or regressor: {0}".format(self.model.estimator)
            raise Exception(err)

        # Check which validation strategy is to be used, if any
        # For an explanation of cross validation in scikit-learn see: http://scikit-learn.org/stable/modules/cross_validation.html#multimetric-cross-validation
        if self.model.cv > 0:
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
        
        # Construct the preprocessor
        prep = Preprocessor(self.model.features_df, scale_hashed=self.model.scale_hashed, scale_vectors=self.model.scale_vectors,\
        missing=self.model.missing, scaler=self.model.scaler, logfile=self.logfile, **self.model.scaler_kwargs)
        
        # Construct a sklearn pipeline
        self.model.pipe = Pipeline([('preprocessor', prep)])

        if self.model.dim_reduction:
            # Construct the dimensionality reduction object
            reduction = self.decomposers[self.model.reduction](**self.model.dim_reduction_args)
            
            # Include dimensionality reduction in the sklearn pipeline
            self.model.pipe.steps.insert(1, ('reduction', reduction))
            self.model.estimation_step = 2
        else:
            self.model.estimation_step = 1      
        
        # Try assuming the pipeline involves a grid search
        try:
            # Construct an estimator
            estimator = self.algorithms[self.model.estimator]()

            # Prepare the grid search using the previously set parameter grid
            grid_search = GridSearchCV(estimator=estimator, param_grid=self.model.param_grid, **self.model.grid_search_args)
            
            # Add grid search to the sklearn pipeline
            self.model.pipe.steps.append(('grid_search', grid_search))

            if self.model.validation == "k-fold":
                # Perform K-fold cross validation
                self._cross_validate()

            # Fit the training data to the pipeline
            self.model.pipe.fit(self.X_train, self.y_train.values.ravel())

            # Get the best parameters and the cross validation results
            grid_search = self.model.pipe.steps[self.model.estimation_step][1]
            self.model.best_params = grid_search.best_params_
            self.model.cv_results = grid_search.cv_results_

            # Get the best estimator to add to the final pipeline
            estimator = grid_search.best_estimator_

            # Update the pipeline with the best estimator
            self.model.pipe.steps[self.model.estimation_step] = ('estimator', estimator)

        except AttributeError:
            # Construct an estimator
            estimator = self.algorithms[self.model.estimator](**self.model.estimator_kwargs)

            # Add the estimator to the sklearn pipeline
            self.model.pipe.steps.append(('estimator', estimator))  

            if self.model.validation == "k-fold":
                # Prepare key word arguments for the estimator step in the pipeline
                # fit_params = {'estimator__' + k: v for k, v in self.model.estimator_kwargs.items()}

                # Perform K-fold cross validation
                self._cross_validate()

            # Fit the training data to the pipeline
            self.model.pipe.fit(self.X_train, self.y_train.values.ravel())
        
        if self.model.validation == "hold-out":       
            # Evaluate the model using the test data            
            self.calculate_metrics(caller="internal")
        
        if self.model.calc_feature_importances:
            # Select the dataset for calculating importances
            if self.model.validation == "hold-out":
                X = self.X_test
                y = self.y_test
            else:
                X = train_test_df
                y = target_df.values.ravel()
            
            # Calculate model agnostic feature importances
            self._calc_importances(X = X, y = y)

        # Persist the model to disk
        self.model = self.model.save(self.model.name, self.path, self.model.compress)
        
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
        
    # STAGE 3 : Allow for larger datasets by using partial fitting methods avaialble with some sklearn algorithmns
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
        
        # Construct a sklearn pipeline
        self.model.pipe = Pipeline([('preprocessor', prep)])

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
    
    def calculate_metrics(self, caller="external"):
        """
        Return key metrics based on a test dataset.
        Metrics returned for a classifier are: accuracy, precision, recall, fscore, support
        Metrics returned for a regressor are: r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score
        """
        
        # If the function call was made externally, process the request
        if caller == "external":
            # Open an existing model and get the training & test dataset and targets based on the request
            self.X_test, self.y_test = self._get_model_and_data()            

        # Get predictions based on the samples
        self.y_pred = self.model.pipe.predict(self.X_test)
        
        # Flatten the y_test DataFrame
        self.y_test = self.y_test.values.ravel()
                    
        # Test the accuracy of the model using the test data
        self.model.score = self.model.pipe.score(self.X_test, self.y_test)
        
        # Try getting the metric_args from the model
        try:
            metric_args = self.model.metric_args
        except AttributeError: 
            metric_args = {}
               
        if self.model.estimator_type == "classifier":
            labels = self.model.pipe.steps[self.model.estimation_step][1].classes_
            
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
            metrics_df.loc["overall", "accuracy"] = self.model.score
            # Finalize the structure of the result DataFrame
            metrics_df.loc[:,"model_name"] = self.model.name
            metrics_df.loc[:,"class"] = metrics_df.index
            metrics_df = metrics_df.loc[:,["model_name", "class", "accuracy", "precision", "recall", "fscore", "support"]]
            
            # Prepare the confusion matrix and add it to the model
            self._prep_confusion_matrix(self.y_test, self.y_pred, labels)
            
        elif self.model.estimator_type == "regressor":
            # Get the r2 score
            metrics_df = pd.DataFrame([[self.model.score]], columns=["r2_score"])
                        
            # Get the mean squared error
            metrics_df.loc[:,"mean_squared_error"] = metrics.mean_squared_error(self.y_test, self.y_pred, **metric_args)
            
            # Get the mean absolute error
            metrics_df.loc[:,"mean_absolute_error"] = metrics.mean_absolute_error(self.y_test, self.y_pred, **metric_args)
            
            # Get the median absolute error
            metrics_df.loc[:,"median_absolute_error"] = metrics.median_absolute_error(self.y_test, self.y_pred)
            
            # Get the explained variance score
            metrics_df.loc[:,"explained_variance_score"] = metrics.explained_variance_score(self.y_test, self.y_pred, **metric_args)
            
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
        self.X = utils.convert_types(self.X, self.model.features_df)
        
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
                    s = s + ", {0}: {1:.3f}".format(self.model.pipe.steps[self.model.estimation_step][1].classes_[i], b)
                    i = i + 1
                probabilities.append(s[2:])
            
            self.y = probabilities
                
        else:
            # Predict y for X using the previously fit pipeline
            self.y = self.model.pipe.predict(self.X)
            
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
        features = self.model.features_df["name"].tolist()
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
        elif self.model.validation == "k-fold":
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
        self.model.overwrite = False
        self.model.debug = False
        self.model.test_size = 0.33
        self.model.cv = 0
        self.model.random_state = 42
        self.model.compress = 3
        self.model.retain_data = False
        self.model.scale_hashed = True
        self.model.scale_vectors = True
        self.model.scaler = "StandardScaler"
        self.model.scaler_kwargs = {}
        self.model.missing = "zeros"
        self.model.calc_feature_importances = False
        
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
                    self.exec_params = {"overwrite":self.model.overwrite, "test_size":self.model.test_size,\
                                        "random_state":self.model.random_state, "compress":self.model.compress,\
                                        "retain_data":self.model.retain_data, "debug":self.model.debug}

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
    
    def _get_model_and_data(self, target=True):
        """
        Get samples and targets based on the request and an existing model's feature definitions.
        If target=False, just return the samples.
        """
    
        # Interpret the request data based on the expected row and column structure
        row_template = ['strData', 'strData']
        col_headers = ['model_name', 'n_features']
        
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
        
        # Split the features provided as a string into individual columns
        samples_df = pd.DataFrame([x[1].split("|") for x in self.request_df.values.tolist()],\
                                     columns=self.model.features_df.loc[:,"name"].tolist(),\
                                     index=self.request_df.index)
        
        # Convert the data types based on feature definitions 
        samples_df = utils.convert_types(samples_df, self.model.features_df)
        
        if target:
            # Get the target feature
            # NOTE: This code block will need to be reviewed for multi-label classification
            target_name = self.model.features_df.loc[self.model.features_df["variable_type"] == "target"].index[0]

            # Get the target data
            target_df = samples_df.loc[:,[target_name]]

        # Get the features to be excluded from the model
        exclusions = self.model.features_df['variable_type'].isin(["excluded", "target", "identifier"])
        
        # Update the feature definitions dataframe
        self.model.features_df = self.model.features_df.loc[~exclusions]
        
        # Remove excluded features from the data
        samples_df = samples_df[self.model.features_df.index.tolist()]
        
        if target:
            return [samples_df, target_df]
        else:
            return samples_df

    def _cross_validate(self, fit_params={}):
        """
        Perform K-fold cross validation on the model for the dataset provided in the request.
        """

        if self.model.estimator_type == "classifier":

            # Get unique labels for classification
            labels = np.unique(self.y_train.values)

            # Flatten the true labels for the training data
            y_train = self.y_train.values.ravel()
            
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
        confusion_array = metrics.confusion_matrix(y_test, y_pred).ravel()
        
        # Structure into a DataFrame suitable for Qlik
        result = []
        i = 0
        for t in labels:
            for p in labels:
                result.append([str(t), str(p), confusion_array[i]])
                i = i + 1
        self.model.confusion_matrix = pd.DataFrame(result, columns=["true_label", "pred_label", "count"])
        self.model.confusion_matrix.loc[:,"model_name"] = self.model.name
        self.model.confusion_matrix = self.model.confusion_matrix.loc[:,["model_name", "true_label", "pred_label", "count"]]
    
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
        
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.model.debug:
            self._print_log(5)
            
        # Send table description
        table_header = (('qlik-tabledescription-bin', self.table.SerializeToString()),)
        self.context.send_initial_metadata(table_header)
    
    def _get_model(self):
        """
        Get the model from the class model cache or disk.
        Update the cache if loading from disk.
        Return the model.
        """
        
        if self.model.name in self.__class__.model_cache:
            # Load the model from cache
            self.model = self.__class__.model_cache[self.model.name]
            
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
    
    def _print_log(self, step):
        """
        Output useful information to stdout and the log file if debugging is required.
        :step: Print the corresponding step in the log
        """
        
        if self.logfile is None:
            # Increment log counter for the class. Each instance of the class generates a new log.
            self.__class__.log_no += 1

            # Create a log file for the instance
            # Logs will be stored in ..\logs\SKLearn Log <n>.txt
            self.logfile = os.path.join(os.getcwd(), 'logs', 'SKLearn Log {}.txt'.format(self.log_no))
        
        if step == 1:
            # Output log header
            sys.stdout.write("\nSKLearnForQlik Log: {0} \n\n".format(time.ctime(time.time())))
            
            with open(self.logfile,'w', encoding='utf-8') as f:
                f.write("SKLearnForQlik Log: {0} \n\n".format(time.ctime(time.time())))
                
        elif step == 2:
            # Output the parameters
            sys.stdout.write("Model Name: {0}\n\n".format(self.model.name))
            sys.stdout.write("Execution arguments: {0}\n\n".format(self.exec_params))
            
            try:
                sys.stdout.write("Scaler: {0}, missing: {1}, scale_hashed: {2}, scale_vectors: {3}\n".format(\
                self.model.scaler, self.model.missing,self.model.scale_hashed, self.model.scale_vectors))
                sys.stdout.write("Scaler kwargs: {0}\n\n".format(self.model.scaler_kwargs))
            except AttributeError:
                sys.stdout.write("scale_hashed: {0}, scale_vectors: {1}\n".format(self.model.scale_hashed, self.model.scale_vectors))

            try:
                if self.model.dim_reduction:
                    sys.stdout.write("Reduction: {0}\nReduction kwargs: {1}\n\n".format(self.model.reduction, self.model.dim_reduction_args))
            except AttributeError:
                pass
            
            sys.stdout.write("Estimator: {0}\nEstimator kwargs: {1}\n\n".format(self.model.estimator,\
                                                                                self.model.estimator_kwargs))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("Model Name: {0}\n\n".format(self.model.name))
                f.write("Execution arguments: {0}\n\n".format(self.exec_params))
                
                try:
                    f.write("Scaler: {0}, missing: {1}, scale_hashed: {2}, scale_vectors: {3}\n".format(self.model.scaler,\
                    self.model.missing, self.model.scale_hashed, self.model.scale_vectors))
                    f.write("Scaler kwargs: {0}\n\n".format(self.model.scaler_kwargs))
                except AttributeError:
                    f.write("scale_hashed: {0}, scale_vectors: {1}\n".format(self.model.scale_hashed, self.model.scale_vectors))

                try:
                    if self.model.dim_reduction:
                        f.write("Reduction: {0}\nReduction kwargs: {1}\n\n".format(self.model.reduction, self.model.dim_reduction_args))
                except AttributeError:
                    pass

                f.write("Estimator: {0}\nEstimator kwargs: {1}\n\n".format(self.model.estimator,self.model.estimator_kwargs))
                
        elif step == 3:                    
            # Output the request dataframe
            sys.stdout.write("REQUEST: {0} rows x cols\nSample Data:\n\n".format(self.request_df.shape))
            sys.stdout.write("{0}\n...\n{1}\n\n".format(self.request_df.head().to_string(), self.request_df.tail().to_string()))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("REQUEST: {0} rows x cols\nSample Data:\n\n".format(self.request_df.shape))
                f.write("{0}\n...\n{1}\n\n".format(self.request_df.head().to_string(), self.request_df.tail().to_string()))
        
        elif step == 4:
            # Output the response dataframe/series
            sys.stdout.write("RESPONSE: {0} rows x cols\nSample Data:\n\n".format(self.response.shape))
            sys.stdout.write("{0}\n...\n{1}\n\n".format(self.response.head().to_string(), self.response.tail().to_string()))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("RESPONSE: {0} rows x cols\nSample Data:\n\n".format(self.response.shape))
                f.write("{0}\n...\n{1}\n\n".format(self.response.head().to_string(), self.response.tail().to_string()))
                 
        elif step == 5:
            # Print the table description if the call was made from the load script
            sys.stdout.write("\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table))
            
            # Write the table description to the log file
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table))
        
        elif step == 6:
            # Message when model is loaded from cache
            sys.stdout.write("\nModel {0} loaded from cache.\n\n".format(self.model.name))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("\nModel {0} loaded from cache.\n\n".format(self.model.name))
            
        elif step == 7:
            # Message when model is loaded from disk
            sys.stdout.write("\nModel {0} loaded from disk.\n\n".format(self.model.name))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("\nModel {0} loaded from disk.\n\n".format(self.model.name))
            
        elif step == 8:
            # Message when cache is updated
            sys.stdout.write("\nCache updated. Models in cache:\n{0}\n\n".format\
                             ([k for k,v in self.__class__.model_cache.items()]))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("\nCache updated. Models in cache:\n{0}\n\n".format([k for k,v in self.__class__.model_cache.items()]))
        
        elif step == 9:
            # Output when a parameter grid is set up
            sys.stdout.write("Model Name: {0}, Estimator: {1}\n\nGrid Search Arguments: {2}\n\nParameter Grid: {3}\n\n".\
            format(self.model.name, self.model.estimator, self.model.grid_search_args, self.model.param_grid))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("Model Name: {0}, Estimator: {1}\n\nGrid Search Arguments: {2}\n\nParameter Grid: {3}\n\n".\
                format(self.model.name, self.model.estimator, self.model.grid_search_args, self.model.param_grid))
    
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