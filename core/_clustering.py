import os
import sys
import time
import string
import numpy as np
import pandas as pd
import hdbscan
import _utils as utils
import ServerSideExtension_pb2 as SSE

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

class HDBSCANForQlik:
    """
    A class to implement the HDBSCAN clustering library for Qlik.
    """
    
    # Counter used to name log files for instances of the class
    log_no = 0
    
    def __init__(self, request, variant="standard"):
        """
        Class initializer.
        :param request: an iterable sequence of RowData
        :Sets up the input data frame and parameters based on the request
        """
        
        # Additional information is printed to the terminal and logs if the paramater debug = true
        if self.debug == 'true':
            self._print_log(1)
        
        # Set the request and variant variables for this object instance
        self.request = request
        self.variant = variant       
        
        if variant == "two_dims":
            row_template = ['strData', 'strData', 'numData', 'strData']
            col_headers = ['dim1', 'dim2', 'measure', 'kwargs']
        elif variant == "lat_long":
            row_template = ['strData', 'numData', 'numData', 'strData']
            col_headers = ['dim1', 'lat', 'long', 'kwargs']
        else:
            row_template = ['strData', 'strData', 'strData']
            col_headers = ['dim1', 'measures', 'kwargs']
        
        # Create a Pandas Data Frame for the request data
        self.request_df = utils.request_df(request, row_template, col_headers)
        
        # Handle null value rows in the request dataset
        self.NaT_df = self.request_df.loc[self.request_df.dim1.isnull()].copy()
        
        # If such a row exists it will be sliced off and then added back to the response
        if len(self.NaT_df) > 0:
            self.request_df = self.request_df.loc[self.request_df.dim1.notnull()]               
        
        # Get additional arguments from the 'kwargs' column in the request data
        # Arguments should take the form of a comma separated string: 'arg1=value1, arg2=value2'
        kwargs = self.request_df.loc[0, 'kwargs']
        self._set_params(kwargs)
        
        # Set up an input Data Frame, excluding the arguments column
        self.input_df = self.request_df.loc[:, self.request_df.difference(['kwargs'])]
        
        # For the two_dims variant we pivot the data to change dim2 into columns
        if variant == "two_dims":
            self.input_df = self.input_df.pivot(index='dim1', columns='dim2')
        # For the other two variants we set the index as the 'dim1' column
        else:
            self.input_df = self.input_df.set_index('dim1')
            
            # For the standard variant we split the measures string into multiple columns and make these values numeric
            if variant == "standard":
                self.input_df = pd.DataFrame([s.split(';') for r in self.input_df.values for s in r], index=self.input_df.index) 
                self.input_df = self.input_df.apply(pd.to_numeric, errors='coerce')
            
        if self.debug == 'true':
            self._print_log(2)
    
    def cluster(self):
        """
        Calculate clusters using the HDBSCAN library.
        """
                
        # Instantiate a HDSCAN object and fit the input data frame:
        self.clusterer = hdbscan.HDBSCAN(**self.hdbscan_kwargs)
        
        self.clusterer.fit(self.input_df)
             
        # Prepare the output
        self.clusters = pd.Series(getattr(self.clusterer, self.result_type))
        
        if self.debug == 'true':
            self._print_log(4)
        
        return self.clusters
    
    def _set_params(self, kwargs):
        """
        Set input parameters based on the request.
        Parameters implemented for the HDBSCAN() function are:  
        Additional parameters used are: scaler, debug
        """
        
        # Set the row count in the original request
        self.request_row_count = len(self.request_df) + len(self.NaT_df)
        
        # Set default values which will be used if arguments are not passed
        self.result_type = 'labels_'
        self.scaler = 'robust'
        self.debug = 'false'
        self.algorithm = None
        self.metric = None
        self.min_cluster_size = None
        self.min_samples = None
        self.p = None
        self.alpha = None
        self.cluster_selection_method = None
        self.allow_single_cluster = None
        self.match_reference_implementation = None
        
        # Set optional parameters
        
        # If the key word arguments were included in the request, get the parameters and values
        if len(kwargs) > 0:
            
            # The parameter and values are transformed into key value pairs
            args = kwargs.translate(str.maketrans('', '', string.whitespace)).split(",")
            self.kwargs = dict([arg.split("=") for arg in args])
            
            # Make sure the key words are in lower case
            self.kwargs = {k.lower(): v for k, v in self.kwargs.items()}
            
            # Set the return type 
            # Valid values are: labels, probabilities, cluster_persistence, outlier_scores
            if 'return' in self.kwargs:
                self.result_type = self.kwargs['return'].lower() + '_'
            
            # Set the standardization strategy for the data
            # Valid values are: standard, minmax, maxabs, robust, quantile
            if 'scale' in self.kwargs:
                self.scaler = self.kwargs['scale'].lower()
            
            # Set the debug option for generating execution logs
            # Valid values are: true, false
            if 'debug' in self.kwargs:
                self.debug = self.kwargs['debug'].lower()
            
            # Set optional parameters for teh HDBSCAN algorithmn
            # For documentation see here: https://hdbscan.readthedocs.io/en/latest/api.html#id20
            
            # Options are: best, generic, prims_kdtree, prims_balltree, boruvka_kdtree, boruvka_balltree
            # Default is 'best'.
            if 'algorithm' in self.kwargs:
                self.algorithm = self.kwargs['algorithm'].lower()
            
            # The metric to use when calculating distance between instances in a feature array.
            # More information here: https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html#what-about-different-metrics
            # Default is 'euclidean'.
            if 'metric' in self.kwargs:
                self.metric = self.kwargs['metric'].lower()
            
            # The minimum size of clusters. 
            # The default value is 5.
            if 'min_cluster_size' in self.kwargs:
                self.min_cluster_size = int(self.kwargs['min_cluster_size'])
                
            # The number of samples in a neighbourhood for a point to be considered a core point.
            if 'min_samples' in self.kwargs:
                self.min_samples = int(self.kwargs['min_samples'])
            
            # p value to use if using the minkowski metric.
            if 'p' in self.kwargs:
                self.p = int(self.kwargs['p'])
            
            # A distance scaling parameter as used in robust single linkage.
            if 'alpha' in self.kwargs:
                self.alpha = float(self.kwargs['alpha'])
            
            # The method used to select clusters from the condensed tree.
            # Options are: eom, leaf.
            if 'cluster_selection_method' in self.kwargs:
                self.cluster_selection_method = self.kwargs['cluster_selection_method'].lower()
            
            # By default HDBSCAN* will not produce a single cluster.
            # Setting this to True will override this and allow single cluster results.
            if 'allow_single_cluster' in self.kwargs:
                self.allow_single_cluster = bool(self.kwargs['allow_single_cluster'])
            
            # There exist some interpretational differences between this HDBSCAN implementation 
            # and the original authors reference implementation in Java. 
            # Note that there is a performance cost for setting this to True.
            if 'match_reference_implementation' in self.kwargs:
                self.match_reference_implementation = bool(self.kwargs['match_reference_implementation'])
        
        # Create dictionary of arguments for the HDBSCAN() function
        self.hdbscan_kwargs = {}
        
        # Populate the parameters in the dictionary:
        
        if self.algorithm is not None:
            self.hdbscan_kwargs['algorithm'] = self.algorithm
        
        if self.metric is not None:
            self.hdbscan_kwargs['metric'] = self.metric
        
        if self.min_cluster_size is not None:
            self.hdbscan_kwargs['min_cluster_size'] = self.min_cluster_size
        
        if self.min_samples is not None:
            self.hdbscan_kwargs['min_samples'] = self.min_samples
        
        if self.p is not None:
            self.hdbscan_kwargs['p'] = self.p
        
        if self.alpha is not None:
            self.hdbscan_kwargs['alpha'] = self.alpha
        
        if self.cluster_selection_method is not None:
            self.hdbscan_kwargs['cluster_selection_method'] = self.cluster_selection_method
        
        if self.allow_single_cluster is not None:
            self.hdbscan_kwargs['allow_single_cluster'] = self.allow_single_cluster
        
        if self.match_reference_implementation is not None:
            self.hdbscan_kwargs['match_reference_implementation'] = self.match_reference_implementation
        
    def _print_log(self, step):
        """
        Output useful information to stdout and the log file if debugging is required.
        step: Print the corresponding step in the log
        """
        
        if step == 1:
            # Increment log counter for the class. Each instance of the class generates a new log.
            self.__class__.log_no += 1
             
            # Create a log file for the instance
            # Logs will be stored in ..\logs\Cluster Log <n>.txt
            self.logfile = os.path.join(os.getcwd(), 'logs', 'Cluster Log {}.txt'.format(self.log_no))
            
            # Output log header
            sys.stdout.write("HDBSCANForQlik Log: {0} \n\n".format(time.ctime(time.time())))
            with open(self.logfile,'w') as f:
                f.write("HDBSCANForQlik Log: {0} \n\n".format(time.ctime(time.time())))
        
        elif step == 2:
            # Output the request and input data frames to the terminal
            sys.stdout.write("Additional parameters: {0}\n\n".format(self.kwargs))
            sys.stdout.write("HDBSCAN parameters: {0}\n\n".format(self.hdbscan_kwargs))
            sys.stdout.write("REQUEST DATA FRAME: {0} rows x cols\n\n".format(self.request_df.shape))
            sys.stdout.write("{0} \n\n".format(self.request_df.to_string()))
            if len(self.NaT_df) > 0:
                sys.stdout.write("REQUEST NULL VALUES DATA FRAME: {0} rows x cols\n\n".format(self.NaT_df.shape))
                sys.stdout.write("{0} \n\n".format(self.NaT_df.to_string()))
            sys.stdout.write("INPUT DATA FRAME: {0} rows x cols\n\n".format(self.input_df.shape))
            sys.stdout.write("{} \n\n".format(self.input_df.to_string()))
                        
            # Output the request and input data frames to the log file 
            with open(self.logfile,'a') as f:
                f.write("Additional parameters: {0}\n\n".format(self.kwargs))
                f.write("HDBSCAN parameters: {0}\n\n".format(self.hdbscan_kwargs))
                f.write("REQUEST DATA FRAME: {0} rows x cols\n\n".format(self.request_df.shape))
                f.write("{0} \n\n".format(self.request_df.to_string()))
                if len(self.NaT_df) > 0:
                    f.write("REQUEST NULL VALUES DATA FRAME: {0} rows x cols\n\n".format(self.NaT_df.shape))
                    f.write("{0} \n\n".format(self.NaT_df.to_string()))
                f.write("INPUT DATA FRAME: {0} rows x cols\n\n".format(self.input_df.shape))
                f.write("{0} \n\n".format(self.input_df.to_string()))
        
        elif step == 3:         
            # Print the output series to the terminal
            sys.stdout.write("\nCLUSTERING RESULTS: \n\n")
            sys.stdout.write("\n{0}\n\n".format(self.clusters.to_string()))
            
            # Write the output data frame and returned series to the log file
            with open(self.logfile,'a') as f:
                f.stdout.write("\nCLUSTERING RESULTS: \n\n")
                f.stdout.write("\n{0}\n\n".format(self.clusters.to_string()))
    
