import argparse
import json
import logging
import logging.config
import os
import sys
import time
from concurrent import futures

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

import ServerSideExtension_pb2 as SSE
import grpc

# Import libraries for added functions
import numpy as np
import pandas as pd
import _utils as utils
from _prophet_forecast import ProphetForQlik
from _clustering import HDBSCANForQlik

# Set the default port for this SSE Extension
_DEFAULT_PORT = '50055'

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_MINFLOAT = float('-inf')


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
            1: '_cluster_by_dim',
            2: '_cluster_geo',
            3: '_correlation',
            4: '_correlation',
            5: '_prophet',
            6: '_prophet',
            7: '_prophet',
            8: '_prophet_seasonality'
        }

    """
    Implementation of added functions.
    """
    
    @staticmethod
    def _cluster(request, context):
        """
        Look for clusters within a dimension using the given features. Scalar function.
        This variant takes in one dimension and a string of features.
        :param request: an iterable sequence of RowData
        :param context:
        :return: the clustering classification for each row
        :Qlik expression examples:
        :<AAI Connection Name>.Cluster(dimension, sum(value1) & ';' & sum(value2), 'scaler=standard')
        """
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
                       
        # Create an instance of the HDBSCANForQlik class
        # This will take the request data from Qlik and prepare it for clustering
        clusterer = HDBSCANForQlik(request_list, context)
        
        # Calculate the clusters and store in a Pandas series (or DataFrame in the case of a load script call)
        clusters = clusterer.scan()
        
        if isinstance(clusters, pd.DataFrame):
            # If the output is a dataframe the columns returned will be ['dim1', 'result']
            response_rows = pd.DataFrame(columns=clusters.columns)
            
            # Values in these columns are converted to type SSE.Dual
            response_rows.loc[:,'dim1'] = clusters.loc[:,'dim1'].apply(lambda s: iter([SSE.Dual(strData=s)]))
            response_rows.loc[:,'result'] = clusters.loc[:,'result'].apply(lambda n: iter([SSE.Dual(numData=n)]))
        else:
            # Values in the response object are converted to type SSE.Dual
            response_rows = clusters.apply(lambda n: iter([SSE.Dual(numData=n)]))
        
        # Values are then structured as SSE.Rows
        # The response is then converted to a list
        response_rows = response_rows.apply(lambda duals: SSE.Row(duals=duals)).values.tolist()        
        
        # Iterate over bundled rows
        for request_rows in request_list:
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)
    
    @staticmethod
    def _cluster_by_dim(request, context):
        """
        Look for clusters within a dimension using the given features. Scalar function.
        This variant can only be used in the load script, not in chart expressions.
        It takes in two dimensions and one measure.
        Features for the clustering are produced by pivoting the data for the second dimension.
        """
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
                       
        # Create an instance of the HDBSCANForQlik class
        # This will take the request data from Qlik and prepare it for clustering
        clusterer = HDBSCANForQlik(request_list, context, variant="two_dims")
        
        # Calculate the clusters and store in a Pandas series (or DataFrame in the case of a load script call)
        clusters = clusterer.scan()
        
        if isinstance(clusters, pd.DataFrame):
            # If the output is a dataframe the columns returned will be ['dim1', 'result']
            response_rows = pd.DataFrame(columns=clusters.columns)
            
            # Values in these columns are converted to type SSE.Dual
            response_rows.loc[:,'dim1'] = clusters.loc[:,'dim1'].apply(lambda s: iter([SSE.Dual(strData=s)]))
            response_rows.loc[:,'result'] = clusters.loc[:,'result'].apply(lambda n: iter([SSE.Dual(numData=n)]))
        else:
            # Values in the response object are converted to type SSE.Dual
            response_rows = clusters.apply(lambda n: iter([SSE.Dual(numData=n)]))
        
        # Values are then structured as SSE.Rows
        # The response is then converted to a list
        response_rows = response_rows.apply(lambda duals: SSE.Row(duals=duals)).values.tolist()        
        
        # Iterate over bundled rows
        for request_rows in request_list:
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)
    
    @staticmethod
    def _cluster_geo(request, context):
        """
        Look for clusters using geospatial coordinates. Scalar function.
        This variant can only be used when latitude and longitude fields are avaialble.
        It takes in one dimension, the latitude and longitude.
        """
        # Get a list from the generator object so that it can be iterated over multiple times
        request_list = [request_rows for request_rows in request]
                       
        # Create an instance of the HDBSCANForQlik class
        # This will take the request data from Qlik and prepare it for clustering
        clusterer = HDBSCANForQlik(request_list, context, variant="lat_long")
        
        # Calculate the clusters and store in a Pandas series (or DataFrame in the case of a load script call)
        clusters = clusterer.scan()
        
        if isinstance(clusters, pd.DataFrame):
            # If the output is a dataframe the columns returned will be ['dim1', 'result']
            response_rows = pd.DataFrame(columns=clusters.columns)
            
            # Values in these columns are converted to type SSE.Dual
            response_rows.loc[:,'dim1'] = clusters.loc[:,'dim1'].apply(lambda s: iter([SSE.Dual(strData=s)]))
            response_rows.loc[:,'result'] = clusters.loc[:,'result'].apply(lambda n: iter([SSE.Dual(numData=n)]))
        else:
            # Values in the response object are converted to type SSE.Dual
            response_rows = clusters.apply(lambda n: iter([SSE.Dual(numData=n)]))
        
        # Values are then structured as SSE.Rows
        # The response is then converted to a list
        response_rows = response_rows.apply(lambda duals: SSE.Row(duals=duals)).values.tolist()   
        
        # Iterate over bundled rows
        for request_rows in request_list:
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)
    
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
                    df = pd.DataFrame({'x': [utils._string_to_float(d) for d in x], \
                                       'y': [utils._string_to_float(d) for d in y]})
                
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
        :Additional parameters used are: return, take_log, debug
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
        predictor = ProphetForQlik(request_list)
        
        # Calculate the forecast and store in a Pandas series
        forecast = predictor.predict()
        
        # Values in the series are converted to type SSE.Dual
        response_rows = forecast.apply(lambda result: iter([SSE.Dual(numData=result)]))
        
        # Values in the series are converted to type SSE.Row
        # The series is then converted to a list
        response_rows = response_rows.apply(lambda duals: SSE.Row(duals=duals)).tolist()        
        
        # Iterate over bundled rows
        for request_rows in request_list:
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)     
    
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
        predictor = ProphetForQlik.init_seasonality(request_list)
        
        # Calculate the forecast and store in a Pandas series
        forecast = predictor.predict()
        
        # Values in the series are converted to type SSE.Dual
        response_rows = forecast.apply(lambda result: iter([SSE.Dual(numData=result)]))
        
        # Values in the series are converted to type SSE.Row
        # The series is then converted to a list
        response_rows = response_rows.apply(lambda duals: SSE.Row(duals=duals)).tolist()        
        
        # Iterate over bundled rows
        for request_rows in request_list:
            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)  
    
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
    
    """
    Implementation of rpc functions.
    """

    def GetCapabilities(self, request, context):
        """
        Get capabilities.
        Note that either request or context is used in the implementation of this method, but still added as
        parameters. The reason is that gRPC always sends both when making a function call and therefore we must include
        them to avoid error messages regarding too many parameters provided from the client.
        :param request: the request, not used in this method.
        :param context: the context, not used in this method.
        :return: the capabilities.
        """
        logging.info('GetCapabilities')

        # Create an instance of the Capabilities grpc message
        # Enable(or disable) script evaluation
        # Set values for pluginIdentifier and pluginVersion
        capabilities = SSE.Capabilities(allowScript=False,
                                        pluginIdentifier='NAF Python Toolbox',
                                        pluginVersion='v1.2.0')

        # If user defined functions supported, add the definitions to the message
        with open(self.function_definitions) as json_file:
            # Iterate over each function definition and add data to the Capabilities grpc message
            for definition in json.load(json_file)['Functions']:
                function = capabilities.functions.add()
                function.name = definition['Name']
                function.functionId = definition['Id']
                function.functionType = definition['Type']
                function.returnType = definition['ReturnType']

                # Retrieve name and type of each parameter
                for param_name, param_type in sorted(definition['Params'].items()):
                    function.params.add(name=param_name, dataType=param_type)

                logging.info('Adding to capabilities: {}({})'.format(function.name,
                                                                     [p.name for p in function.params]))

        return capabilities

    def ExecuteFunction(self, request_iterator, context):
        """
        Call corresponding function based on function id sent in header.
        :param request_iterator: an iterable sequence of RowData.
        :param context: the context.
        :return: an iterable sequence of RowData.
        """
        # Retrieve function id
        func_id = self._get_function_id(context)
        logging.info('ExecuteFunction (functionId: {})'.format(func_id))

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
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
