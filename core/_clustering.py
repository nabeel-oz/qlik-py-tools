import os
import sys
import time
import string
import locale
import warnings
import numpy as np
import pandas as pd
import _utils as utils
import ServerSideExtension_pb2 as SSE

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import hdbscan

# Add Generated folder to module path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

class HDBSCANForQlik:
    """
    A class to implement the HDBSCAN clustering library for Qlik.
    """
    
    # Counter used to name log files for instances of the class
    log_no = 0
    
    def __init__(self, request, context, variant="standard"):
        """
        Class initializer.
        :param request: an iterable sequence of RowData
        :param context:
        :param variant: a string to indicate the request format
        :Sets up the input data frame and parameters based on the request
        """
               
        # Set the request, context and variant variables for this object instance
        self.request = request
        self.context = context
        self.variant = variant
        
        if variant == "two_dims":
            row_template = ['strData', 'strData', 'numData', 'strData']
            col_headers = ['key', 'dim', 'measure', 'kwargs']
        elif variant == "lat_long":
            row_template = ['strData', 'numData', 'numData', 'strData']
            col_headers = ['key', 'lat', 'long', 'kwargs']
        else:
            row_template = ['strData', 'strData', 'strData']
            col_headers = ['key', 'measures', 'kwargs']
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(request, row_template, col_headers)
        
        # Handle null value rows in the request dataset
        self.NaN_df = self.request_df.loc[self.request_df['key'].str.len() == 0].copy()
        
        # If null rows exist they will be sliced off and then added back to the response
        if len(self.NaN_df) > 0:
            self.request_df = self.request_df.loc[self.request_df['key'].str.len() != 0]               
        
        # Get additional arguments from the 'kwargs' column in the request data
        # Arguments should take the form of a comma separated string: 'arg1=value1, arg2=value2'
        kwargs = self.request_df.loc[0, 'kwargs']
        self._set_params(kwargs)
        
        # Additional information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            # Increment log counter for the class. Each instance of the class generates a new log.
            self.__class__.log_no += 1
             
            # Create a log file for the instance
            # Logs will be stored in ..\logs\Cluster Log <n>.txt
            self.logfile = os.path.join(os.getcwd(), 'logs', 'Cluster Log {}.txt'.format(self.log_no))
            
            self._print_log(1)
        
        # Set up an input Data Frame, excluding the arguments column
        self.input_df = self.request_df.loc[:, self.request_df.columns.difference(['kwargs'])]
               
        # For the two_dims variant we pivot the data to change dim into columns and with key as the index
        if variant == "two_dims":
            self.input_df = self.input_df.pivot(index='key', columns='dim')
        # For the other two variants we also set the index as the 'key' column
        else:
            self.input_df = self.input_df.set_index('key')
               
            # For the standard variant we split the measures string into multiple columns and make the values numeric
            if variant == "standard":
                self.input_df = pd.DataFrame([s.split(';') for r in self.input_df.values for s in r], index=self.input_df.index)
        
                # Convert strings to numbers using locale settings
                self.input_df = self.input_df.applymap(lambda s: utils.atof(s) if s else np.NaN)
        
        # Finally we prepare the data for the clustering algorithm:
        
        # If scaling does not need to be applied, we just fill in missing values
        if self.scaler == "none":
            self.input_df = utils.fillna(self.input_df, method=self.missing)
        # Otherwise we apply strategies for both filling missing values and then scaling the data
        else:
            self.input_df = utils.scale(self.input_df, missing=self.missing, scaler=self.scaler, **self.scaler_kwargs)
        
        # For the lat_long variant we do some additional transformations
        if self.variant == "lat_long":
            # The input values are converted to radians
            self.input_df = self.input_df.apply(np.radians)
        
        if self.debug:
            self._print_log(2)
    
    def scan(self):
        """
        Calculate clusters using the HDBSCAN library.
        """
                
        # Instantiate a HDSCAN object and fit the input data frame:
        self.clusterer = hdbscan.HDBSCAN(**self.hdbscan_kwargs)
        
        # Handle exceptions raised by the fit method with some grace
        try:
            # Scan for clusters
            self.clusterer.fit(self.input_df)
            
            # Prepare the output Data Frame
            self.response = pd.DataFrame(getattr(self.clusterer, self.result_type), index=self.input_df.index, columns=['result'])
            
        except ValueError as e:
            # Prepare output Data Frame if clusters can't be generated
            self.response = pd.DataFrame([-1 for i in range(len(self.request_df))], index=self.input_df.index, columns=['result'])
            
            # Print error message
            self._print_exception('ValueError when scanning for clusters', e)
        
        self.response['key'] = self.input_df.index
        self.response = self.response.loc[:, ['key', 'result']]
        
        # Add the null value rows back to the response
        self.response = self.response.append(pd.DataFrame([('\x00', np.NaN) for i in range(len(self.NaN_df))],\
                                                          columns=self.response.columns))
        
        if self.debug:
            self._print_log(3)
        
        # If the function was called through the load script we return a Data Frame
        if self.load_script:
            self._send_table_description()
            
            return self.response
            
        # If the function was called through a chart expression we return a Series
        else:
            return self.response.loc[:,'result']
    
    def _set_params(self, kwargs):
        """
        Set input parameters based on the request.
        :
        :Parameters implemented for the HDBSCAN() function are: algorithm, metric, min_cluster_size, min_samples,
        :p, alpha, cluster_selection_method, allow_single_cluster, match_reference_implementation.
        :More information here: https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan
        :
        :Scaler types implemented for preprocessing data are: StandardScaler, MinMaxScaler, MaxAbsScaler,
        :RobustScaler and QuantileTransformer.
        :More information here: http://scikit-learn.org/stable/modules/preprocessing.html
        :
        :Additional parameters used are: load_script, return, missing, scaler, debug
        """
        
        # Set the row count in the original request
        self.request_row_count = len(self.request_df) + len(self.NaN_df)
        
        # Set default values which will be used if arguments are not passed
        
        # SSE parameters:
        self.load_script = False
        self.result_type = 'labels_'
        self.missing = 'zeros'
        self.scaler = 'robust'
        self.debug = False
        # HDBSCAN parameters:
        self.algorithm = None
        self.metric = None
        self.min_cluster_size = None
        self.min_samples = None
        self.p = None
        self.alpha = None
        self.cluster_selection_method = None
        self.allow_single_cluster = None
        self.match_reference_implementation = None
        # Standard scaler parameters:
        self.with_mean = None
        self.with_std = None
        # MinMaxScaler scaler parameters:
        self.feature_range = None
        # Robust scaler parameters:
        self.with_centering = None
        self.with_scaling = None
        self.quantile_range = None
        # Quantile Transformer parameters:
        self.n_quantiles = None
        self.output_distribution = None
        self.ignore_implicit_zeros = None
        self.subsample = None
        self.random_state = None
        
        # Adjust default options if variant is two_dims
        if self.variant == "two_dims":
            self.load_script = True
        
        # Adjust default options if variant is lat_long
        elif self.variant == "lat_long":
            self.scaler = "none"
            self.metric = "haversine"
        
        # Set optional parameters
        
        # If the key word arguments were included in the request, get the parameters and values
        if len(kwargs) > 0:
            
            # The parameter and values are transformed into key value pairs
            args = kwargs.translate(str.maketrans('', '', string.whitespace)).split(",")
            self.kwargs = dict([arg.split("=") for arg in args])
            
            # Make sure the key words are in lower case
            self.kwargs = {k.lower(): v for k, v in self.kwargs.items()}
            
            # Set the load_script parameter to determine the output format 
            # Set to 'true' if calling the functions from the load script in the Qlik app
            if 'load_script' in self.kwargs:
                self.load_script = 'true' == self.kwargs['load_script'].lower()
            
            # Set the return type 
            # Valid values are: labels, probabilities, cluster_persistence, outlier_scores
            if 'return' in self.kwargs:
                self.result_type = self.kwargs['return'].lower() + '_'
            
            # Set the strategy for missing data
            # Valid values are: zeros, mean, median, mode
            if 'missing' in self.kwargs:
                self.missing = self.kwargs['missing'].lower()
            
            # Set the standardization strategy for the data
            # Valid values are: standard, minmax, maxabs, robust, quantile, none
            if 'scaler' in self.kwargs:
                self.scaler = self.kwargs['scaler'].lower()
            
            # Set the debug option for generating execution logs
            # Valid values are: true, false
            if 'debug' in self.kwargs:
                self.debug = 'true' == self.kwargs['debug'].lower()
            
            # Set optional parameters for the HDBSCAN algorithmn
            # For documentation see here: https://hdbscan.readthedocs.io/en/latest/api.html#id20
            
            # Options are: best, generic, prims_kdtree, prims_balltree, boruvka_kdtree, boruvka_balltree
            # Default is 'best'.
            if 'algorithm' in self.kwargs:
                self.algorithm = self.kwargs['algorithm'].lower()
            
            # The metric to use when calculating distance between instances in a feature array.
            # More information here: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html#what-about-different-metrics
            # And here: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
            # Default is 'euclidean' for 'standard' and 'two_dims' variants, and 'haversine' for the lat_long variant.
            if 'metric' in self.kwargs:
                self.metric = self.kwargs['metric'].lower()
            
            # The minimum size of clusters. 
            # The default value is 5.
            if 'min_cluster_size' in self.kwargs:
                self.min_cluster_size = utils.atoi(self.kwargs['min_cluster_size'])
                
            # The number of samples in a neighbourhood for a point to be considered a core point.
            if 'min_samples' in self.kwargs:
                self.min_samples = utils.atoi(self.kwargs['min_samples'])
            
            # p value to use if using the minkowski metric.
            if 'p' in self.kwargs:
                self.p = utils.atoi(self.kwargs['p'])
            
            # A distance scaling parameter as used in robust single linkage.
            if 'alpha' in self.kwargs:
                self.alpha = utils.atof(self.kwargs['alpha'])
            
            # The method used to select clusters from the condensed tree.
            # Options are: eom, leaf.
            if 'cluster_selection_method' in self.kwargs:
                self.cluster_selection_method = self.kwargs['cluster_selection_method'].lower()
            
            # By default HDBSCAN* will not produce a single cluster.
            # Setting this to True will override this and allow single cluster results.
            if 'allow_single_cluster' in self.kwargs:
                self.allow_single_cluster = 'true' == self.kwargs['allow_single_cluster'].lower()
            
            # There exist some interpretational differences between this HDBSCAN implementation 
            # and the original authors reference implementation in Java. 
            # Note that there is a performance cost for setting this to True.
            if 'match_reference_implementation' in self.kwargs:
                self.match_reference_implementation = 'true' == self.kwargs['match_reference_implementation']
            
            # Set optional parameters for the scaler functions
            
            # Parameters for the Standard scaler
            # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
            if self.scaler == 'standard':
                if 'with_mean' in self.kwargs:
                    self.with_mean = 'true' == self.kwargs['with_mean'].lower()
                if 'with_std' in self.kwargs:
                    self.with_std = 'true' == self.kwargs['with_std'].lower()
            
            # Parameters for the MinMax scaler
            # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            if self.scaler == 'minmax':
                if 'feature_range' in self.kwargs:
                    self.feature_range = ''.join(c for c in self.kwargs['feature_range'] if c not in '()').split(';')
                    self.feature_range = (utils.atoi(self.feature_range[0]),utils.atoi(self.feature_range[1]))
            
            # Parameters for the Robust scaler
            # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
            if self.scaler == 'robust':
                if 'with_centering' in self.kwargs:
                    self.with_centering = 'true' == self.kwargs['with_centering'].lower()
                if 'with_scaling' in self.kwargs:
                    self.with_scaling = 'true' == self.kwargs['with_scaling'].lower()
                if 'quantile_range' in self.kwargs:
                    self.quantile_range = ''.join(c for c in self.kwargs['quantile_range'] if c not in '()').split(';')
                    self.quantile_range = (utils.atof(self.quantile_range[0]),utils.atof(self.quantile_range[1]))
            
            # Parameters for the Quantile Transformer
            # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
            if self.scaler == 'quantile':
                if 'n_quantiles' in self.kwargs:
                    self.n_quantiles = utils.atoi(self.kwargs['n_quantiles'])
                if 'output_distribution' in self.kwargs:
                    self.output_distribution = self.kwargs['output_distribution'].lower()
                if 'ignore_implicit_zeros' in self.kwargs:
                    self.ignore_implicit_zeros = 'true' == self.kwargs['ignore_implicit_zeros'].lower()
                if 'subsample' in self.kwargs:
                    self.subsample = utils.atoi(self.kwargs['subsample'])
                if 'random_state' in self.kwargs:
                    self.random_state = utils.atoi(self.kwargs['random_state'])
        
        # Set up a list of possible key word arguments for the HDBSCAN() function
        hdbscan_params = ['algorithm', 'metric', 'min_cluster_size', 'min_samples', 'p', 'alpha',\
                          'cluster_selection_method', 'allow_single_cluster', 'match_reference_implementation']
        
        # Create dictionary of key word arguments for the HDBSCAN() function
        self.hdbscan_kwargs = self._populate_dict(hdbscan_params)
               
        # Set up a list of possible key word arguments for the sklearn preprocessing functions
        scaler_params = ['with_mean', 'with_std', 'feature_range', 'with_centering', 'with_scaling',\
                        'quantile_range', 'n_quantiles', 'output_distribution', 'ignore_implicit_zeros',\
                        'subsample', 'random_state']
        
        # Create dictionary of key word arguments for the scaler functions
        self.scaler_kwargs = self._populate_dict(scaler_params)
        
    def _populate_dict(self, params):
        """
        Populate a dictionary based on a list of parameters. 
        The parameters should already exist in this object.
        """
        
        output_dict = {}
        
        for prop in params:
            if getattr(self, prop) is not None:
                output_dict[prop] = getattr(self, prop)
        
        return output_dict
    
    def _send_table_description(self):
        """
        Send the table description to Qlik as meta data.
        Only used when the SSE is called from the Qlik load script.
        """
        
        # Set up the table description to send as metadata to Qlik
        self.table = SSE.TableDescription()
        self.table.name = "ClusteringResults"
        self.table.numberOfRows = len(self.response)

        # Set up fields for the table
        self.table.fields.add(name="key")
        self.table.fields.add(name=self.result_type[:-1], dataType=1)
        
        if self.debug:
            self._print_log(4)
        
        # Send table description
        table_header = (('qlik-tabledescription-bin', self.table.SerializeToString()),)
        self.context.send_initial_metadata(table_header)
    
    def _print_log(self, step):
        """
        Output useful information to stdout and the log file if debugging is required.
        :step: Print the corresponding step in the log
        """
        
        if step == 1:
            # Output log header
            sys.stdout.write("\nHDBSCANForQlik Log: {0} \n\n".format(time.ctime(time.time())))
            with open(self.logfile,'w') as f:
                f.write("HDBSCANForQlik Log: {0} \n\n".format(time.ctime(time.time())))
        
        elif step == 2:
            # Output the request and input data frames to the terminal
            sys.stdout.write("Key word arguments: {0}\n\n".format(self.kwargs))
            sys.stdout.write("HDBSCAN parameters: {0}\n\n".format(self.hdbscan_kwargs))
            if self.scaler == "none":
                sys.stdout.write("No scaling applied\n\n")
            else:
                sys.stdout.write("{0} scaler parameters: {1}\n\n".format(self.scaler.capitalize(), self.scaler_kwargs))
            sys.stdout.write("REQUEST DATA FRAME: {0} rows x cols\n\n".format(self.request_df.shape))
            sys.stdout.write("{0} \n\n".format(self.request_df.to_string()))
            if len(self.NaN_df) > 0:
                sys.stdout.write("REQUEST NULL VALUES DATA FRAME: {0} rows x cols\n\n".format(self.NaN_df.shape))
                sys.stdout.write("{0} \n\n".format(self.NaN_df.to_string()))
            sys.stdout.write("INPUT DATA FRAME: {0} rows x cols\n\n".format(self.input_df.shape))
            sys.stdout.write("{0} \n\n".format(self.input_df.to_string()))
                        
            # Output the request and input data frames to the log file 
            with open(self.logfile,'a') as f:
                f.write("Key word arguments: {0}\n\n".format(self.kwargs))
                f.write("HDBSCAN parameters: {0}\n\n".format(self.hdbscan_kwargs))
                if self.scaler == "none":
                    f.write("No scaling applied\n\n")
                else:
                    f.write("{0} scaler parameters: {1}\n\n".format(self.scaler.capitalize(), self.scaler_kwargs))
                f.write("REQUEST DATA FRAME: {0} rows x cols\n\n".format(self.request_df.shape))
                f.write("{0} \n\n".format(self.request_df.to_string()))
                if len(self.NaN_df) > 0:
                    f.write("REQUEST NULL VALUES DATA FRAME: {0} rows x cols\n\n".format(self.NaN_df.shape))
                    f.write("{0} \n\n".format(self.NaN_df.to_string()))
                f.write("INPUT DATA FRAME: {0} rows x cols\n\n".format(self.input_df.shape))
                f.write("{0} \n\n".format(self.input_df.to_string()))
        
        elif step == 3:         
            # Print the output series to the terminal
            sys.stdout.write("\nCLUSTERING RESULTS: {0} rows x cols\n\n".format(self.response.shape))
            sys.stdout.write("{0} \n\n".format(self.response.to_string()))
            
            # Write the output data frame and returned series to the log file
            with open(self.logfile,'a') as f:
                f.write("\nCLUSTERING RESULTS: {0} rows x cols\n\n".format(self.response.shape))
                f.write("{0} \n\n".format(self.response.to_string()))
        
        elif step == 4:
            # Print the table description if the call was made from the load script
            sys.stdout.write("\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table))
            
            # Write the table description to the log file
            with open(self.logfile,'a') as f:
                f.write("\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table))
    
    def _print_exception(self, s, e):
        """
        Output exception message to stdout and also to the log file if debugging is required.
        :s: A description for the error
        :e: The exception
        """
        
        # Output exception message
        sys.stdout.write("\n{0}: {1} \n\n".format(s, e))
        
        if self.debug:
            with open(self.logfile,'a') as f:
                f.write("\n{0}: {1} \n\n".format(s, e))
            
    
