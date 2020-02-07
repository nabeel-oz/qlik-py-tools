import argparse
import json
import logging
import logging.config
import re
import os
import sys
import time
import locale
import warnings
from concurrent import futures

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

import ServerSideExtension_pb2 as SSE
import grpc

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Import libraries for added functions
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import _utils as utils
from _prophet import ProphetForQlik
from _clustering import HDBSCANForQlik
from _sklearn import SKLearnForQlik
from _spacy import SpaCyForQlik
from _common import CommonFunction

# Set the default port for this SSE Extension
_DEFAULT_PORT = '50055'

# Set the maximum message length for gRPC in bytes
MAX_MESSAGE_LENGTH = 10 * 1024 * 1024

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_MINFLOAT = float('-inf')

# Set the locale for number formats based on user settings
locale.setlocale(locale.LC_NUMERIC, '')

class ExtensionService(SSE.ConnectorServicer):
    """
    A SSE-plugin to provide Python data science functions for Qlik.
    """

    def __init__(self, funcdef_file):
        """
        Class initializer.
        :param funcdef_file: a function definition JSON file
        """
        self._function_definitions = funcdef_file
        os.makedirs('logs', exist_ok=True)
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logger.config')
        logging.config.fileConfig(log_file)
        logging.info('Logging enabled')

    @property
    def function_definitions(self):
        """
        :return: json file with function definitions
        """
        return self._function_definitions

    @property
    def functions(self):
        """
        :return: Mapping of function id and implementation
        """
        return {
            0: '_cluster',
            1: '_cluster',
            2: '_cluster',
            3: '_correlation',
            4: '_correlation',
            5: '_prophet',
            6: '_prophet',
            7: '_prophet',
            8: '_prophet_seasonality',
            9: '_sklearn',
            10: '_sklearn',
            11: '_sklearn',
            12: '_sklearn',
            13: '_sklearn',
            14: '_sklearn',
            15: '_sklearn',
            16: '_sklearn',
            17: '_sklearn',
            18: '_sklearn',
            19: '_sklearn',
            20: '_sklearn',
            21: '_sklearn',
            22: '_sklearn',
            23: '_sklearn',
            24: '_sklearn',
            25: '_sklearn',
            26: '_sklearn',
            27: '_sklearn',
            28: '_sklearn',
            29: '_sklearn',
            30: '_spacy',
            31: '_spacy',
            32: '_spacy',
            33: '_misc',
            34: '_sklearn',
            35: '_sklearn',
            36: '_sklearn',
            37: '_sklearn',
            38: '_sklearn',
            39: '_sklearn',
            40: '_prophet',
            41: '_prophet_seasonality',
            42: '_sklearn',
            43: '_misc',
            44: '_misc',
            45: '_misc'
        }

    """
    Implementation of added functions.
    """
    
    @staticmethod
    def _cluster(request, context):
        """
        Look for clusters within a dimension using the given features. Scalar function.
        Three variants are implemented:
        :0: one dimension and a string of numeric features.
        :1: two dimensions and one measure. This will only work when used in the Qlik load script.
        :2: one dimension, the latitude and longitude.
        :
        :param request: an iterable sequence of RowData
        :param context:
        :return: the clustering classification for each row
        :Qlik expression examples:
        :<AAI Connection Name>.Cluster(dimension, sum(value1) & ';' & sum(value2), 'scaler=standard')
        """
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
        
        # Get the function id from the header to determine the variant being called
        function = ExtensionService._get_function_id(context)
               
        if function == 0:
            # The standard variant takes in one dimension and a string of semi-colon separated features.
            variant = "standard"
        elif function == 1:
            # In the second variant features for clustering are produced by pivoting the measure for the second dimension.
            variant = "two_dims"
        elif function == 2:
            # The third variant if for finding clusters in geospatial coordinates.
            variant = "lat_long"
        
        # Create an instance of the HDBSCANForQlik class
        # This will take the request data from Qlik and prepare it for clustering
        clusterer = HDBSCANForQlik(request_list, context, variant=variant)
        
        # Calculate the clusters and store in a Pandas series (or DataFrame in the case of a load script call)
        clusters = clusterer.scan()
        
        # Check if the response is a DataFrame. 
        # This occurs when the load_script=true argument is passed in the Qlik expression.
        response_is_df = isinstance(clusters, pd.DataFrame)
        
        # Convert the response to a list of rows
        clusters = clusters.values.tolist()
        
        # We convert values to type SSE.Dual, and group columns into a iterable
        if response_is_df:
            response_rows = [iter([SSE.Dual(strData=row[0]),SSE.Dual(numData=row[1])]) for row in clusters]
        else:
            response_rows = [iter([SSE.Dual(numData=row)]) for row in clusters]
        
        # Values are then structured as SSE.Rows
        response_rows = [SSE.Row(duals=duals) for duals in response_rows]

        # Get the number of bundles in the request
        num_request_bundles = len(request_list)

        # Get the number of rows in the response
        num_rows = len(response_rows) 

        # Calculate the number of rows to send per bundle
        if num_rows >= num_request_bundles:
            rows_per_bundle = num_rows//num_request_bundles
        else:
            rows_per_bundle = num_rows

        # Stream response as BundledRows
        for i in range(0, num_rows, rows_per_bundle):
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows[i : i + rows_per_bundle])
    
    @staticmethod
    def _correlation(request, context):
        """
        Calculate the correlation coefficient for two columns. Scalar function.
        :param request: an iterable sequence of RowData
        :param context:
        :return: the correlation coefficient for each row
        :Qlik expression examples:
        :<AAI Connection Name>.Pearson('1;NA;3;4;5;6.9', ';11;12;;14;')
        :<AAI Connection Name>.Correlation('1;NA;3;4;5;6.9', ';11;12;;14;', 'pearson')
        :Possible values for the third argument are 'pearson', 'kendall' or 'spearman'
        """
        # Iterate over bundled rows
        for request_rows in request:
            response_rows = []
            
            # Set to True for additional info in terminal and log file
            debug = False
            
            if debug:
                # Create a log file for the 
                logfile = os.path.join(os.getcwd(), 'logs', 'Correlation Log.txt')
                
                sys.stdout.write("Function Call: {0} \n\n".format(time.ctime(time.time())))
                with open(logfile,'a') as f:
                    f.write("Function Call: {0} \n\n".format(time.ctime(time.time())))
            
            # Iterating over rows
            for row in request_rows.rows:
                # Retrieve the value of the parameters
                # Two or Three columns are sent from the client, hence the length of params will be 2 or 3
                params = [col.strData for col in row.duals]
                
                if debug:
                    sys.stdout.write("\nPARAMETERS:\n\n{0}\n".format("\n\n".join(str(x) for x in params)))
                    with open(logfile,'a') as f:
                        f.write("\nPARAMETERS:\n\n{0}\n".format("\n\n".join(str(x) for x in params)))
                
                # Create lists for the two series
                x = params[0].split(";")
                y = params[1].split(";")

                # Set the correlation type based on the third argument. 
                # Default is Pearson if the arg is missing.
                try:
                    corr_type = params[2].lower()
                except IndexError:
                    corr_type = 'pearson'
                
                if debug:
                    sys.stdout.write("\n\nx ({0:d} data points):\n{1}\n".format(len(x), " ".join(str(v) for v in x)))
                    sys.stdout.write("\ny ({0:d} data points):\n{1}\n".format(len(y), " ".join(str(v) for v in y)))
                    sys.stdout.write("\nCorrelation Type: {0}\n\n".format(corr_type))
                    
                    with open(logfile,'a') as f:
                        f.write("\n\nx ({0:d} data points):\n{1}\n".format(len(x), " ".join(str(v) for v in x)))
                        f.write("\ny ({0:d} data points):\n{1}\n".format(len(y), " ".join(str(v) for v in y)))
                        f.write("\nCorrelation Type: {0}\n\n".format(corr_type))
                
                # Check that the lists are of equal length
                if len(x) == len(y) and len(x) > 0:
                    # Create a Pandas data frame using the lists
                    df = pd.DataFrame({'x': [utils.atof(d) for d in x], \
                                       'y': [utils.atof(d) for d in y]})
                
                    # Calculate the correlation matrix for the two series in the data frame
                    corr_matrix = df.corr(method=corr_type)
                    
                    if debug:
                        sys.stdout.write("\n\nCorrelation Matrix:\n{}\n".format(corr_matrix.to_string()))
                        with open(logfile,'a') as f:
                            f.write("\n\nCorrelation Matrix:\n{}\n".format(corr_matrix.to_string()))
                    
                    # Prepare the result
                    if corr_matrix.size > 1:
                        result = corr_matrix.iloc[0,1]
                    else:
                        result = None
                else:
                    result = None               

                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])

                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)
    
    @staticmethod
    def _prophet(request, context):
        """
        Provide a timeseries forecast using Facebook's Prophet library. Scalar function.
        :param request: an iterable sequence of RowData
        :param context: not used for now
        :return: the forecasted value for each row
        :
        :Qlik expression example:
        :<AAI Connection Name>.Prophet(MonthStartDate, sum(Value), 'return=yhat, freq=MS, debug=true')
        :The third argument in the Qlik expression is a string of parameters. 
        :This should take the form of a comma separated string:
        :e.g 'return=yhat, freq=MS, debug=true' or 'return=yhat_upper, freq=MS'
        :
        :<AAI Connection Name>.Prophet_Holidays(ForecastDate, sum(Value), Holiday, 'return=yhat, freq=D, debug=true')
        :In the holidays variant the third argument is a field containing the holiday name or NULL for each row.
        :
        :Parameters accepted for the Prophet() function are: cap, floor, changepoint_prior_scale, interval_width, 
        :lower_window, upper_window 
        :
        :Parameters accepted for the make_future_dataframe() function are: freq
        :
        :For more information on these parameters go here: https://facebook.github.io/prophet/docs/quick_start.html
        :
        :Additional parameters used are: return, take_log, debug, load_script
        :
        :cap = 1000 : A logistic growth model can be defined using cap and floor. Values should be double or integer
        :changepoint_prior_scale = 0.05 : Decrease if the trend changes are being overfit, increase for underfit
        :interval_width = 0.08 : Set the width of the uncertainty intervals
        :lower_window = 1 : Only used with holidays. Extend the holiday by certain no. of days prior to the date.
        :upper_window = 1 : Only used with holidays. Extend the holiday by certain no. of days after the date.
        :freq = MS : The frequency of the time series. e.g. MS for Month Start. See the possible options here:
        :          : http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        :return = yhat : Any of the options in the forecast result. You can see these options with debug=true
        :              : yhat, yhat_upper, yhat_lower : Forecast, upper and lower limits
        :              : y_then_yhat, y_then_yhat_upper, y_then_yhat_lower : Return forecast only for forecast periods
        :              : trend, trend_upper, trend_lower : Trend component of the timeseries
        :              : seasonal, seasonal_upper, seasonal_lower: Seasonal component of the timeseries 
        :take_log = false : Apply logarithm to the values before the forecast. Default is true
        :debug = true : Print execution information to the terminal and logs in ..\logs\Prophet Log <n>.txt
        """
        
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
        
        # Calculate timings for the components of the forecasting
        # The results will be stored in ..\logs\Prophet Performance Log.txt
        # The request_list line above is not timed as the generator can only be iterated once
        # ProphetForQlik.timeit(request_list)
                       
        # Create an instance of the ProphetForQlik class
        # This will take the request data from Qlik and prepare it for forecasting
        predictor = ProphetForQlik(request_list, context)
        
        # Calculate the forecast and store in a Pandas series
        forecast = predictor.predict()  
        
        # Check if the response is a DataFrame. 
        # This occurs when the load_script=true argument is passed in the Qlik expression.
        response_is_df = isinstance(forecast, pd.DataFrame)   

        # Set the data types of the output
        if response_is_df:
            dtypes = []
            for dt in forecast.dtypes:
                dtypes.append('num' if is_numeric_dtype(dt) else 'str')
        else:
            dtypes = ['num']

        # Get the response as SSE.Rows
        response_rows = utils.get_response_rows(forecast.values.tolist(), dtypes) 

        # Get the number of bundles in the request
        num_request_bundles = len(request_list)

        # Get the number of rows in the response
        num_rows = len(response_rows) 

        # Calculate the number of rows to send per bundle
        if num_rows >= num_request_bundles:
            rows_per_bundle = num_rows//num_request_bundles
        else:
            rows_per_bundle = num_rows

        # Stream response as BundledRows
        for i in range(0, num_rows, rows_per_bundle):
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows[i : i + rows_per_bundle])  
    
    @staticmethod
    def _prophet_seasonality(request, context):
        """
        Provide the seasonality component of the Prophet timeseries forecast. Scalar function.
        :param request: an iterable sequence of RowData
        :param context: not used for now
        :return: the forecasted value for each row
        :
        :Qlik expression example:
        :<AAI Connection Name>.Prophet_Seasonality(Month, $(vConcatSeries), $(vHolidays), 'seasonality=yearly, freq=MS, debug=true')
        :The fourth argument in the Qlik expression is a string of parameters. 
        :This should take the form of a comma separated string:
        :e.g 'seasonality=yearly, freq=MS, debug=true' or 'seasonality=weekly, freq=D'
        :
        :Parameters accepted for the Prophet() function are: cap, floor, changepoint_prior_scale, interval_width, 
        :lower_window, upper_window 
        :
        :Parameters accepted for the make_future_dataframe() function are: freq
        :
        :For more information on these parameters go here: https://facebook.github.io/prophet/docs/quick_start.html
        :
        :Additional parameters used are: return, take_log, debug
        :
        :cap = 1000 : A logistic growth model can be defined using cap and floor. Values should be double or integer
        :changepoint_prior_scale = 0.05 : Decrease if the trend changes are being overfit, increase for underfit
        :interval_width = 0.08 : Set the width of the uncertainty intervals
        :lower_window = -1 : Only used with holidays. Extend the holiday by certain no. of days prior to the date.
        :upper_window = 1 : Only used with holidays. Extend the holiday by certain no. of days after the date.
        :freq = MS : The frequency of the time series. e.g. MS for Month Start. See the possible options here:
        :          : http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        :return = yhat : Any of the options in the forecast result. You can see these options with debug=true 
        :              : yhat, yhat_upper, yhat_lower : Forecast, upper and lower limits
        :              : y_then_yhat, y_then_yhat_upper, y_then_yhat_lower : Return forecast only for forecast periods
        :              : trend, trend_upper, trend_lower : Trend component of the timeseries
        :              : seasonal, seasonal_upper, seasonal_lower: Seasonal component of the timeseries 
        :take_log = false : Apply logarithm to the values before the forecast. Default is true
        :debug = true : Print exexution information to the terminal and logs in ..\logs\Prophet Log <n>.txt
        """
        
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
                              
        # Create an instance of the ProphetForQlik class
        # This will take the request data from Qlik and prepare it for forecasting
        predictor = ProphetForQlik.init_seasonality(request_list, context)
        
        # Calculate the forecast and store in a Pandas series
        forecast = predictor.predict()
        
        # Values in the series are converted to type SSE.Dual
        response_rows = forecast.apply(lambda result: iter([SSE.Dual(numData=result)]))
        
        # Values in the series are converted to type SSE.Row
        # The series is then converted to a list
        response_rows = response_rows.apply(lambda duals: SSE.Row(duals=duals)).tolist()        
        
        # Get the number of bundles in the request
        num_request_bundles = len(request_list)

        # Get the number of rows in the response
        num_rows = len(response_rows) 

        # Calculate the number of rows to send per bundle
        if num_rows >= num_request_bundles:
            rows_per_bundle = num_rows//num_request_bundles
        else:
            rows_per_bundle = num_rows

        # Stream response as BundledRows
        for i in range(0, num_rows, rows_per_bundle):
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows[i : i + rows_per_bundle]) 
    
    @staticmethod
    def _sklearn(request, context):
        """
        Execute functions for the sklearn machine learning library.
        :param request: an iterable sequence of RowData
        :param context:
        :return: Refer to comments below as the response depends on the function called
        """
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
        
        # Get the function id from the header to determine the variant being called
        function = ExtensionService._get_function_id(context)
        
        # Create an instance of the SKLearnForQlik class
        model = SKLearnForQlik(request_list, context)
        
        # Call the function based on the mapping in functions.json
        # The if conditions are grouped based on similar output structure
        if function in (9, 10, 21, 24, 34):    
            if function == 9:
                # Set up the model and save to disk
                response = model.setup()
            elif function == 21:
                # Set up a model with specific metric and dimensionality reduction arguments and save to disk
                response = model.setup(advanced=True)
            elif function == 10:
                # Set feature definitions for an existing model
                response = model.set_features()
            elif function == 24:
                # Set a parameter grid for hyperparameter optimization
                response = model.set_param_grid()
            elif function == 34:
                # Setup the architecture for a Keras model
                response = model.keras_setup()
            
            dtypes = ["str", "str", "str"]
        
        elif function == 11:
            # Return the feature definitions for an existing model
            response = model.get_features()
            dtypes = ["str", "num", "str", "str", "str", "str", "str"]
        
        elif function == 12:
            # Train and Test an existing model, saving the sklearn pipeline for further predictions
            response = model.fit()
            dtypes = ["str", "str", "str", "str", "num"]
        
        elif function in (14, 16, 19, 20, 27, 36):
            if function == 14:
                # Provide predictions in a chart expression based on an existing model
                response = model.predict(load_script=False)
            elif function == 16:
                # Provide predictions probabilities in a chart expression based on an existing model
                response = model.predict(load_script=False, variant="predict_proba")
            elif function == 19:
                # Get a list of models based on a search string
                response = model.list_models()
            elif function == 20:
                # Get a string that can be evaluated to get the features expression for the predict function
                response = model.get_features_expression()
            elif function == 27:
                # Get labels for clustering
                response = model.fit_transform(load_script=False)
            elif function == 36:
                # Get sequence predictions from Keras
                response = model.sequence_predict(load_script=False)
            elif function == 38:
                # Get sequence prediction probabilities from Keras
                response = model.sequence_predict(load_script=False, variant="predict_proba")
            
            dtypes = ["str"]
            
        elif function in (15, 17, 28, 37):
            if function == 15:
                # Provide predictions in the load script based on an existing model
                response = model.predict(load_script=True)    
            elif function == 17:
                # Provide prediction probabilities in the load script based on an existing model
                response = model.predict(load_script=True, variant="predict_proba")
            elif function == 28:
                # Provide labels for clustering
                response = model.fit_transform(load_script=True)
            elif function == 37:
                # Get sequence predictions from Keras
                response = model.sequence_predict(load_script=True)
            elif function == 39:
                # Get sequence prediction probabilities from Keras
                response = model.sequence_predict(load_script=True, variant="predict_proba")

            dtypes = ["str", "str", "str"]
        
        elif function in (18, 22, 42):
            if function == 18:
                response = model.get_metrics()
            elif function == 22:
                response = model.calculate_metrics()
            elif function == 42:
                response = model.calculate_metrics(ordered_data=True)
            
            # Check whether the metrics are for a classifier or regressor and whether they come from cross validation or hold-out testing
            if "accuracy_std" in response.columns:
                estimator_type = "classifier_cv"
            elif "accuracy" in response.columns:
                estimator_type = "classifier"
            elif "r2_score_std" in response.columns:
                estimator_type = "regressor_cv"
            elif "r2_score" in response.columns:
                estimator_type = "regressor"
            
            # We convert values to type SSE.Dual, and group columns into a iterable
            if estimator_type == "classifier_cv":
                dtypes = ["str", "str", "num", "num", "num", "num", "num", "num", "num", "num"]
            elif estimator_type == "classifier":
                dtypes = ["str", "str", "num", "num", "num", "num", "num"]
            elif estimator_type == "regressor_cv":
                dtypes = ["str", "num", "num", "num", "num", "num", "num", "num", "num", "num", "num"]
            elif estimator_type == "regressor":
                dtypes = ["str", "num", "num", "num", "num", "num"]
        
        elif function == 23:
            # Get the confusion matrix for the classifier
            response = model.get_confusion_matrix()
            if response.shape[1] == 4:
                dtypes = ["str", "str", "str", "num"]
            else:
                dtypes = ["str", "num", "num", "num", "num", "num"]
        
        elif function == 25:
            # Get the best parameters based on a grid search cross validation
            response = model.get_best_params()
            dtypes = ["str", "str"]
        
        elif function == 26:
            # Provide results from dimensionality reduction
            response = model.fit_transform(load_script=True)
            dtypes = ["str", "str"]

            for i in range(response.shape[1]-2):
                dtypes.append("num")
        
        elif function == 29:
            # Explain the feature importances for the model
            response = model.explain_importances()
            dtypes = ["str", "str", "num"]
        
        elif function == 35:
            # Provide metrics from the training history of a Keras model
            response = model.get_keras_history()
            dtypes = ["str"]

            for i in range(1, response.shape[1]):
                dtypes.append("num")
        
        # Get the response as SSE.Rows
        response_rows = utils.get_response_rows(response.values.tolist(), dtypes) 

        # Get the number of rows in the request
        num_request_bundles = len(request_list)

        # Get the number of rows in the response
        num_rows = len(response_rows) 

        # Calculate the number of rows to send per bundle
        if num_rows >= num_request_bundles:
            rows_per_bundle = len(response_rows)//len(request_list)
        else:
            rows_per_bundle = num_rows

        # Stream response as BundledRows
        for i in range(0, len(response_rows), rows_per_bundle):
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows[i : i + rows_per_bundle])
    
    @staticmethod
    def _spacy(request, context):
        """
        Execute functions for the spaCy natural language processing library.
        :param request: an iterable sequence of RowData
        :param context:
        :return: Refer to comments below as the response depends on the function called
        """
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
        
        # Get the function id from the header to determine the variant being called
        function = ExtensionService._get_function_id(context)
        
        # Create an instance of the SpaCyForQlik class
        model = SpaCyForQlik(request_list, context)
        
        # Call the function based on the mapping in functions.json
        # The if conditions are grouped based on similar output structure
        if function in (30, 31):    
            if function == 30:
                # Get entities from the default model
                response = model.get_entities()
            elif function == 31:
                # Get entities from a named model
                response = model.get_entities(default=False)
            
            # return six columns: key, entity, start, end, type, description
            dtypes = ["str", "str", "num", "num", "str", "str"]
        
        elif function == 32:
            # Retrain a model by supplying texts and labeled entities
            response = model.retrain()

            # return four columns: model_name, subset, metric, value
            dtypes = ["str", "str", "str", "num"]

        # Get the response as SSE.Rows
        response_rows = utils.get_response_rows(response.values.tolist(), dtypes) 

        # Get the number of rows in the request
        num_request_bundles = len(request_list)

        # Get the number of rows in the response
        num_rows = len(response_rows) 

        # Calculate the number of rows to send per bundle
        if num_rows >= num_request_bundles:
            rows_per_bundle = len(response_rows)//len(request_list)
        else:
            rows_per_bundle = num_rows

        # Stream response as BundledRows
        for i in range(0, len(response_rows), rows_per_bundle):
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows[i : i + rows_per_bundle])
    
    @staticmethod
    def _misc(request, context):
        """
        Execute functions that provide common data science capabilities for Qlik.
        :param request: an iterable sequence of RowData
        :param context:
        :return: Refer to comments below as the response depends on the function called
        """
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
        
        # Get the function id from the header to determine the variant being called
        function = ExtensionService._get_function_id(context)
        
        # Create an instance of the CommonFunction class
        handle = CommonFunction(request_list, context)
        
        # Call the function based on the mapping in functions.json
        # The if conditions are grouped based on similar output structure
        if function == 33:    
            # Get entities from the default model
            response = handle.association_rules()
            # return six columns: 'rule', 'rule_lhs', 'rule_rhs', 'support', 'confidence', 'lift'
            dtypes = ["str", "str", "str", "num", "num", "num"]
        elif function == 43:
            # Provide predictions in a chart expression based on an existing model
            response = handle.predict(load_script=False)
            # Return predictions
            dtypes = ["str"]
        elif function == 44:
            # Provide predictions in the load script based on an existing model
            response = handle.predict(load_script=True)
            # Return the model name, keys and predictions
            dtypes = ["str", "str", "str"]
        elif function == 45:
            # Get a string that can be evaluated to get the features expression for the predict function
            response = handle.get_features_expression()
            # Return the feature expression
            dtypes = ["str"]

        # Get the response as SSE.Rows
        response_rows = utils.get_response_rows(response.values.tolist(), dtypes) 

        # Get the number of bundles in the request
        num_request_bundles = len(request_list)

        # Get the number of rows in the response
        num_rows = len(response_rows) 

        # Calculate the number of rows to send per bundle
        if num_rows >= num_request_bundles:
            rows_per_bundle = num_rows//num_request_bundles
        else:
            rows_per_bundle = num_rows

        # Stream response as BundledRows
        for i in range(0, num_rows, rows_per_bundle):
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows[i : i + rows_per_bundle])
    
    @staticmethod
    def _get_function_id(context):
        """
        Retrieve function id from header.
        :param context: context
        :return: function id
        """
        metadata = dict(context.invocation_metadata())
        header = SSE.FunctionRequestHeader()
        header.ParseFromString(metadata['qlik-functionrequestheader-bin'])

        return header.functionId

    def _get_call_info(self, context):
        """
        Retreive useful information for the function call.
        :param context: context
        :return: string containing header info
        """

        # Get metadata for the call from the context
        metadata = dict(context.invocation_metadata())
        
        # Get the function ID
        func_header = SSE.FunctionRequestHeader()
        func_header.ParseFromString(metadata['qlik-functionrequestheader-bin'])
        func_id = func_header.functionId

        # Get the common request header
        common_header = SSE.CommonRequestHeader()
        common_header.ParseFromString(metadata['qlik-commonrequestheader-bin'])

        # Get capabilities
        if not hasattr(self, 'capabilities'):
            self.capabilities = self.GetCapabilities(None, context, log=False)

        # Get the name of the capability called in the function
        capability = [function.name for function in self.capabilities.functions if function.functionId == func_id][0]
                
        # Get the user ID using a regular expression
        match = re.match(r"UserDirectory=(?P<UserDirectory>\w*)\W+UserId=(?P<UserId>\w*)", common_header.userId, re.IGNORECASE)
        if match:
            userId = match.group('UserDirectory') + '/' + match.group('UserId')
        else:
            userId = common_header.userId
        
        # Get the app ID
        appId = common_header.appId
        # Get the call's origin
        peer = context.peer()

        return "{0} - Capability '{1}' called by user {2} from app {3}".format(peer, capability, userId, appId)
    
    """
    Implementation of rpc functions.
    """

    def GetCapabilities(self, request, context, log=True):
        """
        Get capabilities.
        Note that either request or context is used in the implementation of this method, but still added as
        parameters. The reason is that gRPC always sends both when making a function call and therefore we must include
        them to avoid error messages regarding too many parameters provided from the client.
        :param request: the request, not used in this method.
        :param context: the context, not used in this method.
        :return: the capabilities.
        """
        if log:
            logging.info('GetCapabilities')

        # Create an instance of the Capabilities grpc message
        # Enable(or disable) script evaluation
        # Set values for pluginIdentifier and pluginVersion
        self.capabilities = SSE.Capabilities(allowScript=False,
                                        pluginIdentifier='Qlik Python Tools',
                                        pluginVersion='v2.3.0')

        # If user defined functions supported, add the definitions to the message
        with open(self.function_definitions) as json_file:
            # Iterate over each function definition and add data to the Capabilities grpc message
            for definition in json.load(json_file)['Functions']:
                function = self.capabilities.functions.add()
                function.name = definition['Name']
                function.functionId = definition['Id']
                function.functionType = definition['Type']
                function.returnType = definition['ReturnType']

                # Retrieve name and type of each parameter
                for param_name, param_type in sorted(definition['Params'].items()):
                    function.params.add(name=param_name, dataType=param_type)

                if log:
                    logging.info('Adding to capabilities: {}({})'.format(function.name,
                                                                        [p.name for p in function.params]))

        return self.capabilities

    def ExecuteFunction(self, request_iterator, context):
        """
        Call corresponding function based on function id sent in header.
        :param request_iterator: an iterable sequence of RowData.
        :param context: the context.
        :return: an iterable sequence of RowData.
        """
        # Retrieve function id
        func_id = self._get_function_id(context)
        logging.info(self._get_call_info(context))
        logging.info('ExecuteFunction (functionId: {}, {})'.format(func_id, self.functions[func_id]))
        
        return getattr(self, self.functions[func_id])(request_iterator, context)

    """
    Implementation of the Server connecting to gRPC.
    """

    def Serve(self, port, pem_dir):
        """
        Server
        :param port: port to listen on.
        :param pem_dir: Directory including certificates
        :return: None
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),\
        options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
        
        SSE.add_ConnectorServicer_to_server(self, server)

        if pem_dir:
            # Secure connection
            with open(os.path.join(pem_dir, 'sse_server_key.pem'), 'rb') as f:
                private_key = f.read()
            with open(os.path.join(pem_dir, 'sse_server_cert.pem'), 'rb') as f:
                cert_chain = f.read()
            with open(os.path.join(pem_dir, 'root_cert.pem'), 'rb') as f:
                root_cert = f.read()
            credentials = grpc.ssl_server_credentials([(private_key, cert_chain)], root_cert, True)
            server.add_secure_port('[::]:{}'.format(port), credentials)
            logging.info('*** Running server in secure mode on port: {} ***'.format(port))
        else:
            # Insecure connection
            server.add_insecure_port('[::]:{}'.format(port))
            logging.info('*** Running server in insecure mode on port: {} ***'.format(port))

        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)

class AAIException(Exception):
    """
    Custom exception call to pass on information error messages
    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', nargs='?', default=_DEFAULT_PORT)
    parser.add_argument('--pem_dir', nargs='?')
    parser.add_argument('--definition_file', nargs='?', default='functions.json')
    args = parser.parse_args()

    # need to locate the file when script is called from outside it's location dir.
    def_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.definition_file)

    calc = ExtensionService(def_file)
    calc.Serve(args.port, args.pem_dir)
