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

from fbprophet import Prophet, plot

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

class ProphetForQlik:
    """
    A class to provide Facebook Prophet functions for Qlik.
    """
    
    # Counter used to name log files for instances of the class
    log_no = 0
    
    # Dates in Qlik are stored as serial number that equals the number of days since December 30, 1899. 
    # This variable is used in correctly translating dates.
    qlik_cal_start = pd.Timestamp('1899-12-30')
    # This variable denotes the unit of time used in Qlik for numerical representation of datetime values
    qlik_cal_unit = 'D'
    
    def __init__(self, request, context):
        """
        Class initializer.
        :param request: an iterable sequence of RowData
        :Sets up the input data frame and parameters based on the request
        """
        
        # Set the request and context variables for this object instance
        self.request = request
        self.context = context

        # Create a Pandas Data Frame with column ds for the dates and column y for values
        self.request_df = pd.DataFrame([(row.duals[0].numData, row.duals[1].numData) \
                                        for request_rows in self.request \
                                        for row in request_rows.rows], \
                                       columns=['ds','y'])
        
        # Handle null value rows in the request dataset
        self.NaT_df = self.request_df.loc[self.request_df.ds.isnull()].copy()
        
        # If such a row exists it will be sliced off and then added back to the response
        if len(self.NaT_df) > 0:
            self.NaT_df.loc[:,'y'] = 0
            self.request_df = self.request_df.loc[self.request_df.ds.notnull()]               
        
        # Get additional arguments from the third column in the request data
        # Arguments should take the form of a comma separated string: 'arg1=value1, arg2=value2'
        self._set_params()
        
        # Additional information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(1)
        
        # Convert numerical date values to datetime
        self.request_df.loc[:,'ds'] = pd.to_datetime(self.request_df.loc[:,'ds'], unit=self.qlik_cal_unit,
                                                     origin=self.qlik_cal_start)
        
        # If the request contains holidays, prepare a holidays data frame
        if self.has_holidays:
            self._prep_holidays()
        
        # If the request contains additional regressors, add them to a regressors data frame
        if self.has_regressors:
            self._prep_regressors()
        
        # Sort the Request Data Frame based on dates, as Qlik may send unordered data
        self.request_df = self.request_df.sort_values('ds')
        
        # Store the original indexes for re-ordering output later
        self.request_index = self.request_df.loc[:,'ds']
        
        # Add additional regressors to the request data frame
        if self.has_regressors:
            self.request_df = self.request_df.merge(self.regressors_df, how='left', left_index=True, right_index=True)

        # Ignore the placeholder rows which will be filled with forecasted figures later
        self.input_df = self.request_df.iloc[:-self.periods].copy()
        
        # Reset the indexes for the input data frame. 
        # Not doing this interferes with correct ordering of the output from Prophet
        self.input_df = self.input_df.reset_index(drop=True)
        
        # If the input data frame contains less than 2 non-Null rows, prediction is not possible
        if len(self.input_df) - self.input_df.y.isnull().sum() >= 2:
            # If take_log = true take logarithm of relevant input values.
            # This is usually to make the timeseries more stationary
            if self.take_log:
                self.input_df.loc[:,'y'] = np.log(self.input_df.loc[:,'y'])
                
                if self.cap is not None:
                    self.cap = np.log(self.cap)
                
                    if self.floor is not None:
                        self.floor = np.log(self.floor)
            
            # If a logistic growth model is applied add the cap and floor columns to the input data frame
            if self.cap is not None:
                self.input_df.loc[:,'cap'] = self.cap
                
                if self.floor is not None:
                    self.input_df.loc[:,'floor'] = self.floor
            
        if self.debug:
            self._print_log(2)
    
    @classmethod
    def init_seasonality(cls, request, context):
        """
        Alternative initialization method for this class
        Used when the request contains the timeseries as a contatenated string, repeated for every row
        This is used when the number of input data points differs from the output rows required for seasonality plots
        """
                
        # The rows are duplicates in this kind of request, so inputs are simply taken from the first row
        # First we store the correct number of rows to be output.
        request_row_count = len([row for request_rows in request for row in request_rows.rows])
        # The timeseries is accepted as a string from the second column of the first row
        timeseries = request[0].rows[0].duals[1].strData
        # The holidays are taken from the third column of the first row
        holidays = request[0].rows[0].duals[2].strData
        
        # Get the number of columns in the request
        cols = len(request[0].rows[0].duals)

        # If additional regressors are included we extract them from the request as well
        if cols > 4:
            regressors = request[0].rows[0].duals[3].strData
            regressor_args = request[0].rows[0].duals[4]
        
        # The key word arguments are taken from the last column of the first row
        args = request[0].rows[0].duals[cols-1]
        
        # The data may be sent unsorted by Qlik, so we have to store the order to use when sending the results
        sort_order = pd.DataFrame([(row.duals[0].numData, row.duals[0].strData) \
                                        for request_rows in request \
                                        for row in request_rows.rows], \
                                       columns=['seasonality_num', 'seasonality_str'])
        
        # We ignore Null values here as these are handled separately in the response
        sort_order = sort_order.loc[sort_order.seasonality_num.notnull()]
        
        # Re-create the request with ds and y columns
        pairs = timeseries.split(";")
        request_df = pd.DataFrame([p.split(":") for p in pairs], columns=['ds', 'y'])
        
        # Convert strings to numeric values, replace conversion errors with Null values
        request_df = request_df.applymap(lambda s: utils.atof(s) if s else np.NaN)      
        
        # Check if the holidays column is populated
        if len(holidays) > 0:
            # Create a holidays data frame
            pairs = holidays.split(";")
            holiday_df = pd.DataFrame([p.split(":") for p in pairs], columns=['ds', 'holiday'])
            
            # Workaround for Pandas not converting the ds column to floats like it does for request_df
            holiday_df.loc[:,'ds'] = holiday_df.loc[:,'ds'].astype('float64')
            
            # Merge the holidays with the request data frame using column ds as key
            request_df = pd.merge(request_df, holiday_df, on='ds', how='left')
            
            # Replace null values in the holiday column with empty strings
            request_df = request_df.fillna(value={'holiday': ''})
        
        # If additional regressors are included in the request
        if cols > 4:
            # Create a regressors data frame
            pairs = regressors.split(";")
            regressors_df = pd.DataFrame([p.split(":") for p in pairs], columns=['ds', 'regressors'])
            
            # Merge the holidays with the request data frame using column ds as key
            request_df = pd.merge(request_df, regressors_df, on='ds', how='left')
            
            # Replace null values in the holiday column with empty strings
            request_df = request_df.fillna(value={'regressors': ''})

            # Add keyword arguments for the additional regressors to the request data frame as well
            request_df.loc[:, 'regressor_args'] = regressor_args

        # Values in the data frame are converted to type SSE.Dual
        request_df.loc[:,'ds'] = request_df.loc[:,'ds'].apply(lambda result: SSE.Dual(numData=result))
        request_df.loc[:,'y'] = request_df.loc[:,'y'].apply(lambda result: SSE.Dual(numData=result))
        if 'holiday' in request_df.columns:
            request_df.loc[:,'holiday'] = request_df.loc[:,'holiday'].apply(lambda result: SSE.Dual(strData=result))
        if 'regressors' in request_df.columns:
            request_df.loc[:,'regressors'] = request_df.loc[:,'regressors'].apply(lambda result: SSE.Dual(strData=result))
        
        # Add the keyword arguments to the data frame as well, already of type SSE.Dual
        request_df.loc[:, 'args'] = args
        
        # Create the updated request list and convert to SSE data types
        request_list = request_df.values.tolist()
        request_list = [SSE.Row(duals=duals) for duals in request_list]
        updated_request = [SSE.BundledRows(rows=request_list)]
                
        # Call the default initialization method
        instance = ProphetForQlik(updated_request, context)
        
        # Handle null value row in the request dataset
        instance.NaT_df = request_df.loc[request_df.ds.isnull()].copy()
        
        # If such a row exists it will be sliced off and then added back to the response
        if len(instance.NaT_df) > 0:
            instance.NaT_df.loc[:,'y'] = 0         
        
        # Set a property that lets us know this instance was created for seasonality forecasts
        instance.is_seasonality_request = True
        
        # Set a property that lets us know the row count in the original request as this will be different from request_df
        instance.request_row_count = request_row_count
        
        # Update the default result type if this was not passed in arguments
        if instance.result_type == 'yhat':
            instance.result_type = instance.seasonality
        
        if instance.seasonality == 'weekly':
            # For weekly seasonlity the return sort order is based on the day number from 0-6, with 0 being Monday
            instance.sort_order = sort_order.set_index(sort_order.seasonality_num)
        else:
            # Else the return sort order is based on the data frame's index after sorting on the seasonality field
            instance.sort_order = sort_order.sort_values('seasonality_num')
        
        # Return the initialized ProphetForQlik instance
        return instance
    
    def predict(self):
        """
        Calculate forecasted values using the Prophet library.
        """
        
        # If the input data frame contains less than 2 non-Null rows, prediction is not possible
        if len(self.input_df) - self.input_df.y.isnull().sum() <= 2:
            
            if self.debug:
                self._print_log(3)
            
            # A series of null values is returned to avoid an error in Qlik
            return pd.Series([np.NaN for y in range(self.request_row_count)])
        
        # Instantiate a Prophet object and fit the input data frame:
        
        if len(self.prophet_kwargs) > 0:
            self.model = Prophet(**self.prophet_kwargs)
        else:
            self.model = Prophet()
        
        # Add custom seasonalities if defined in the arguments
        if self.name is not None and len(self.add_seasonality_kwargs) > 0:
            self.model.add_seasonality(**self.add_seasonality_kwargs)
        
        # Add additional regressors if defined in the arguments
        if self.has_regressors:
            i=0
            for regressor in self.regressors_df.columns:
                self.model.add_regressor(regressor, **self.regressor_kwargs[i])
                i+=1

        self.model.fit(self.input_df, **self.fit_kwargs)
             
        # Create a data frame for future values
        self.future_df = self.model.make_future_dataframe(**self.make_kwargs)
        
        # If a logistic growth model is applied add the cap and floor columns to the future data frame
        if self.cap is not None:
            self.future_df.loc[:,'cap'] = self.cap
            
            if self.floor is not None:
                self.future_df.loc[:,'floor'] = self.floor
        
        # Add additional regressors to the future data frame
        if self.has_regressors:
            # index_slice = self.regressors_df.shape[0] - self.periods
            for regressor in self.regressors_df.columns:
                self.future_df[regressor] = self.regressors_df.loc[:, regressor]

        if self.debug:
            self._print_log(4)

        # Prepare the forecast
        self._forecast()
                
        # If the function was called through the load script we return a Data Frame
        if self.load_script:            
            # If the response is the seasonality plot we return all seasonality components
            if self.is_seasonality_request:
                # Add an index column to the response
                self.response = self.forecast.reset_index()
            # Otherwise we add dates to the response
            else:
                # Set up the response data frame
                self.response = self.forecast if self.result_type == 'all' else self.forecast.loc[:, ['ds', self.result_type]]
                # Update the ds column as formatted strings
                self.response['ds'] = self.request_df['ds'].dt.strftime('%Y-%m-%d %r')
            
            if self.debug:
                self._print_log(5)
            
            # Send meta data on the response to Qlik
            self._send_table_description()
            
            return self.response
        else:
            if self.debug:
                self._print_log(5)

            return self.forecast.loc[:,self.result_type]
    
    def _set_params(self):
        """
        Set input parameters based on the request.
        Parameters implemented for the Prophet() function are: growth, cap, floor, changepoint_prior_scale, interval_width 
        Parameters implemented for the make_future_dataframe() function are: freq, periods
        Parameters implemented for seasonality are: add_seasonality, seasonality_period, seasonality_fourier, seasonality_prior_scale
        Parameters implemented for holidays are: holidays_prior_scale, lower_window, upper_window
        Additional parameters for seasonlity requests are: weekly_start, yearly_start
        Additional parameters used are: return, take_log, seasonality, debug
        """
        
        # Calculate the forecast periods based on the number of placeholders in the data
        self.periods = utils.count_placeholders(self.request_df.loc[:,'y'])
        
        # Set the row count in the original request
        self.request_row_count = len(self.request_df) + len(self.NaT_df)
        
        # Set default values which will be used if an argument is not passed
        self.load_script = False
        self.result_type = 'yhat'
        self.take_log  = False
        self.seasonality = 'yearly'
        self.seasonality_mode = None
        self.debug = False
        self.freq = 'D'
        self.cap = None
        self.floor = None
        self.growth = None
        self.changepoint_prior_scale = None
        self.interval_width = None
        self.name = None
        self.period = None
        self.fourier_order = None
        self.mode = None
        self.seasonality_prior_scale = None
        self.holidays_prior_scale = None
        self.mcmc_samples = None
        self.seed = None
        self.n_changepoints = None
        self.changepoint_range = None
        self.uncertainty_samples = None
        self.is_seasonality_request = False
        self.weekly_start = 1 # Defaulting to a Monday start for the week as used in Qlik
        self.yearly_start = 0
        self.lower_window = None
        self.upper_window = None
        
        # Set optional parameters
        
        # Check the number of columns in the request to determine whether we have holidays and/or added regressors
        cols = len(self.request[0].rows[0].duals)
        self.has_holidays = False
        self.has_regressors = False

        # If we receive five columns, we expect both holidays and additional regressors
        if cols == 6:
            self.has_regressors = True
        # For a request with four columns, we only expect holidays
        if cols >= 4:
            self.has_holidays = True

        # If there are three or more columns, the last column should contain the key word arguments        
        if cols < 3:
            args = None
        else:
            args = self.request[0].rows[0].duals[cols-1].strData
                
        # If the key word arguments were included in the request, get the parameters and values
        if args is not None:
            
            # The parameter and values are transformed into key value pairs
            args = args.translate(str.maketrans('', '', string.whitespace)).split(",")
            self.kwargs = dict([arg.split("=") for arg in args])
            
            # Make sure the key words are in lower case
            self.kwargs = {k.lower(): v for k, v in self.kwargs.items()}
            
            # Set the load_script parameter to determine the output format 
            # Set to 'true' if calling the functions from the load script in the Qlik app
            if 'load_script' in self.kwargs:
                self.load_script = 'true' == self.kwargs['load_script'].lower()
            
            # Set the return type 
            # Valid values are: yhat, trend, seasonal, seasonalities, all, y_then_yhat, residual. 
            # Add _lower or _upper to the series name to get lower or upper limits.
            # The special case of 'all' returns all output columns from Prophet. This can only be used with 'load_script=true'.
            # 'y_then_yhat' returns actual values for historical periods and forecast values for future periods
            # 'residual' returns y - yhat for historical periods
            if 'return' in self.kwargs:
                self.result_type = self.kwargs['return'].lower()

            # Set a flag to return the seasonality plot instead
            # Only usable through the load script as the result will have a different cardinality to the request
            if 'is_seasonality_request' in self.kwargs:
                self.is_seasonality_request = 'true' == self.kwargs['is_seasonality_request'].lower()
                self.load_script = True
            
            # Set the option to take a logarithm of y values before forecast calculations
            # Valid values are: true, false
            if 'take_log' in self.kwargs:
                self.take_log = 'true' == self.kwargs['take_log'].lower()
                
            # Set the type of seasonlity requested. Used only for seasonality requests
            # Valid values are: yearly, weekly, monthly, holidays
            if 'seasonality' in self.kwargs:
                self.seasonality = self.kwargs['seasonality'].lower()
            
            # Set the seasonlity mode. Useful if the seasonality is not a constant additive factor as assumed by Prophet
            # Valid values are: additive, multiplicative
            if 'seasonality_mode' in self.kwargs:
                self.seasonality_mode = self.kwargs['seasonality_mode'].lower()
            
            # Set the debug option for generating execution logs
            # Valid values are: true, false
            if 'debug' in self.kwargs:
                self.debug = 'true' == self.kwargs['debug'].lower()
            
            # Set the frequency of the timeseries
            # Any valid frequency for pd.date_range, such as 'D' or 'M' 
            # For options see: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
            if 'freq' in self.kwargs:
                self.freq = self.kwargs['freq']
            
            # Set the cap which adds an upper limit at which the forecast will saturate
            # This changes the default linear growth model to a logistic growth model
            if 'cap' in self.kwargs:
                self.cap = utils.atof(self.kwargs['cap'])
                self.growth = 'logistic'
            
                # Set the floor which adds a lower limit at which the forecast will saturate
                # To use a logistic growth trend with a floor, a cap must also be specified
                if 'floor' in self.kwargs:
                    self.floor = utils.atof(self.kwargs['floor'])
            
            # Set the changepoint_prior_scale to adjust the trend flexibility
            # If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), 
            # you can adjust the strength of the sparse prior. 
            # Default value is 0.05. Increasing it will make the trend more flexible.
            if 'changepoint_prior_scale' in self.kwargs:
                self.changepoint_prior_scale = utils.atof(self.kwargs['changepoint_prior_scale'])
            
            # Set the width for the uncertainty intervals
            # Default value is 0.8 (i.e. 80%)
            if 'interval_width' in self.kwargs:
                self.interval_width = utils.atof(self.kwargs['interval_width'])
            
            # Set additional seasonality to be added to the model
            # Default seasonalities are yearly and weekly, as well as daily for sub daily data
            if 'add_seasonality' in self.kwargs:
                self.name = self.kwargs['add_seasonality'].lower()
            
            # Set 'additive' or 'multiplicative' mode for the additional seasonality
            # Default value follows the seasonality_mode parameter
            if 'add_seasonality_mode' in self.kwargs:
                self.mode = self.kwargs['add_seasonality_mode'].lower()
            
            # Set the seasonality period 
            # e.g. 30.5 for 'monthly' seasonality
            if 'seasonality_period' in self.kwargs:
                self.period = utils.atof(self.kwargs['seasonality_period'])
            
            # Set the seasonality fourier terms 
            # Increasing the number of Fourier terms allows the seasonality to fit faster changing cycles, 
            # but can also lead to overfitting
            if 'seasonality_fourier' in self.kwargs:
                self.fourier_order = int(self.kwargs['seasonality_fourier'])
            
            # Set the seasonality prior scale to smooth seasonality effects. 
            # Reducing this parameter dampens seasonal effects
            if 'seasonality_prior_scale' in self.kwargs:
                self.seasonality_prior_scale = utils.atof(self.kwargs['seasonality_prior_scale'])
            
            # Set the holiday prior scale to smooth holiday effects. 
            # Reducing this parameter dampens holiday effects. Default is 10, which provides very little regularization.
            if 'holidays_prior_scale' in self.kwargs:
                self.holidays_prior_scale = utils.atof(self.kwargs['holidays_prior_scale'])

            # Set the number of MCMC samples. 
            # If greater than 0, Prophet will do full Bayesian inference with the specified number of MCMC samples. 
            # If 0, Prophet will do MAP estimation. Default is 0.
            if 'mcmc_samples' in self.kwargs:
                self.mcmc_samples = utils.atoi(self.kwargs['mcmc_samples'])
            
            # Random seed that can be used to control stochasticity. 
            # Used for setting the numpy random seed used in predict and also for pystan when using mcmc_samples>0.
            if 'random_seed' in self.kwargs:
                self.seed = utils.atoi(self.kwargs['random_seed'])

                # Set the random seed for numpy
                np.random.seed(self.seed)

            # Number of potential changepoints to include. Default value is 25.
            # Potential changepoints are selected uniformly from the first `changepoint_range` proportion of the history.
            if 'n_changepoints' in self.kwargs:
                self.n_changepoints = utils.atoi(self.kwargs['n_changepoints'])

            # Proportion of history in which trend changepoints will be estimated. 
            # Defaults to 0.8 for the first 80%.
            if 'changepoint_range' in self.kwargs:
                self.changepoint_range = utils.atof(self.kwargs['changepoint_range'])

            # Number of simulated draws used to estimate uncertainty intervals.
            if 'uncertainty_samples' in self.kwargs:
                self.uncertainty_samples = utils.atoi(self.kwargs['uncertainty_samples'])
            
            # Set the weekly start for 'weekly' seasonality requests 
            # Default week start is 0 which represents Sunday. Add offset as required.
            if 'weekly_start' in self.kwargs:
                self.weekly_start = utils.atoi(self.kwargs['weekly_start'])
            
            # Set the weekly start for 'yearly' seasonality requests 
            # Default week start is 0 which represents 1st of Jan. Add offset as required.
            if 'yearly_start' in self.kwargs:
                self.yearly_start = utils.atoi(self.kwargs['yearly_start'])
            
            # Set a period to extend the holidays by lower_window number of days before the date. 
            # This can be used to extend the holiday effect
            if 'lower_window' in self.kwargs:
                self.lower_window = utils.atoi(self.kwargs['lower_window'])
            
            # Set a period to extend the holidays by upper_window number of days after the date. 
            # This can be used to extend the holiday effect
            if 'upper_window' in self.kwargs:
                self.upper_window = utils.atoi(self.kwargs['upper_window'])
        
        # Create dictionary of arguments for the Prophet(), make_future_dataframe(), add_seasonality() and fit() functions
        self.prophet_kwargs = {}
        self.make_kwargs = {}
        self.add_seasonality_kwargs = {}
        self.fit_kwargs = {}
        
        # Populate the parameters in the corresponding dictionary:
        
        # Set up a list of possible key word arguments for the Prophet() function
        prophet_params = ['seasonality_mode', 'growth', 'changepoint_prior_scale', 'interval_width',\
                          'seasonality_prior_scale', 'holidays_prior_scale', 'mcmc_samples', 'n_changepoints',\
                          'changepoint_range', 'uncertainty_samples']
        
        # Create dictionary of key word arguments for the Prophet() function
        self.prophet_kwargs = self._populate_dict(prophet_params)
        
        # Set up a list of possible key word arguments for the make_future_dataframe() function
        make_params = ['periods', 'freq']
        
        # Create dictionary of key word arguments for the make_future_dataframe() function
        self.make_kwargs = self._populate_dict(make_params)
        
        # Set up a list of possible key word arguments for the add_seasonality() function
        seasonality_params = ['name', 'period', 'fourier_order', 'mode']
        
        # Create dictionary of key word arguments for the add_seasonality() function
        self.add_seasonality_kwargs = self._populate_dict(seasonality_params)

        # Pass the random seed to the fit method if MCMC is being used
        if self.mcmc_samples is not None and self.mcmc_samples > 0:
            # Set up a list of possible key word arguments for the fit() function
            fit_params = ['seed']
            # Create dictionary of key word arguments for the fit() function
            self.fit_kwargs = self._populate_dict(fit_params)
            
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
    
    def _prep_holidays(self):
        """
        Prepare the holidays data frame.
        The request should contain a holiday column which provides the holidays for past and future dates.
        The column provides holiday names, while the ds column provides the holiday's date. 
        Rows without a holiday name are considered non-holidays and not part of the holiday data frame.
        """

        # Create a holidays data frame
        self.holidays_df = pd.DataFrame([(row.duals[0].numData, row.duals[2].strData)\
                                            for request_rows in self.request\
                                            for row in request_rows.rows],\
                                        columns=['ds','holiday'])
        
        # Add upper and lower window for the holidays if applicable
        if self.lower_window is not None:
            self.holidays_df.loc[:, 'lower_window'] = self.lower_window
        if self.upper_window is not None:
            self.holidays_df.loc[:, 'upper_window'] = self.upper_window
                
        # Copy dates from the request_df
        self.holidays_df.loc[:,'ds'] = self.request_df.loc[:,'ds'].copy()
        
        # Remove rows from the holidays data frame where the holiday or ds column is empty
        self.holidays_df = self.holidays_df.loc[self.holidays_df.holiday != '']
        self.holidays_df = self.holidays_df.loc[self.holidays_df.ds.notnull()]

        # If the holidays data frame is empty we don't need to add it to the key word arguments for prophet
        if self.holidays_df.empty:
            self.has_holidays = False
            return
        
        # Make the holidays names lower case to avoid the return argument becoming case sensitive
        self.holidays_df.loc[:,'holiday'] = self.holidays_df.holiday.str.lower()
        # Also remove spaces and apostrophes
        self.holidays_df.loc[:,'holiday'] = self.holidays_df.holiday.str.replace(" ", "_")
        self.holidays_df.loc[:,'holiday'] = self.holidays_df.holiday.str.replace("'", "")
        
        # Sort by the ds column and reset indexes
        self.holidays_df = self.holidays_df.sort_values('ds').reset_index(drop=True)
        
        # Finally add this to the key word argumemnts for Prophet
        self.prophet_kwargs['holidays'] = self.holidays_df
    
    def _prep_regressors(self):
        """
        Parse the request for additional regressors and arguments.
        The regressors are expected as a string of pipe separated values.
        e.g. a single entry with three regressors could be '1.2|200|3'
        
        Arguments for the regressors can be passed in a separate string of keyword arguments.
        The keyword and the value should be separated by equals signs, different keywords by commas, and arguments for different regressors by pipe.
        If a single set of arguments is provided (i.e. no pipe characters are found), we apply the same arguments to all regressors.
        e.g. 'prior_scale=10, mode=additive| mode=multiplicative| mode=multiplicative' for specifying different arguments per regressor
              or 'mode=additive' for using the same arguments for all regressors.

        Returns a data frame with the additional regressors.
        """

        # Create a Pandas Data Frame with additional regressors and their keyword arguments
        self.regressors_df = pd.DataFrame([(row.duals[0].numData, row.duals[3].strData, row.duals[4].strData) \
            for request_rows in self.request \
                for row in request_rows.rows], \
                    columns=['ds', 'regressors', 'kwargs'])
        
        # Handle null value rows in the request dataset
        self.regressors_df = self.regressors_df.loc[self.regressors_df.ds.notnull()]               
        
        # Check if the regressors column is empty
        if len(self.regressors_df.regressors.unique()) == 1:
            # Return without further processing
            self.has_regressors = False
            if self.debug:
                self._print_log(7)
            return None

        # Get the regressor arguments as a string
        arg_string = self.regressors_df.loc[0, 'kwargs']
        
        # Add kwargs for regressors to a list of dictionaries
        self.regressor_kwargs = []
        for kwargs_string in arg_string.replace(' ', '').split('|'):
            if len(kwargs_string) > 0:
                kwargs = {}
                for kv in kwargs_string.split(','):
                    pair = kv.split('=')
                    if 'prior_scale' in pair[0]:
                        pair[1] = utils.atof(pair[1])
                    if 'standardize' in pair[0] and pair[1].lower() != 'auto':
                        pair[1] = 'true' == pair[1].lower()
                    kwargs[pair[0]] = pair[1]
                self.regressor_kwargs.append(kwargs) 

        # Split up the additional regressors into multiple columns
        self.regressors_df = pd.DataFrame(self.regressors_df.regressors.str.split('|', expand=True).values, \
            index=self.regressors_df.index).add_prefix('regressor_')
        
        # Convert the strings to floats
        self.regressors_df = self.regressors_df.applymap(utils.atof)
        
        # Copy dates from the request_df
        self.regressors_df.loc[:,'ds'] = self.request_df.loc[:,'ds'].copy()

        # Sort by the ds column and reset indexes
        self.regressors_df = self.regressors_df.sort_values('ds').reset_index(drop=True).drop(columns=['ds'])

        # If there are no regressor kwargs add empty dictionaries
        if len(self.regressor_kwargs) == 0:
            self.regressor_kwargs = [{} for c in self.regressors_df.columns]
        # If there is just 1 dictionary, replicate it for each regressor
        elif len(self.regressor_kwargs) == 1:
            kwargs = self.regressor_kwargs[0].copy()
            self.regressor_kwargs = [kwargs for c in self.regressors_df.columns]
        elif len(self.regressor_kwargs) != len(self.regressors_df.columns):
            err = "The number of additional regressors does not match the keyword arguments provided for the regressors."
            raise IndexError(err) 

        return self.regressors_df
    
    def _forecast(self):
        """
        Execute the forecast algorithm according to the request type
        """
        
        # If this is a seasonality request, we need to return the relevant seasonlity component
        if self.is_seasonality_request:

            if self.seasonality == 'weekly':
                # Prepare the seasonality data frame
                # Parameter start needs to be any arbitrary week starting on a Sunday
                days = (pd.date_range(start='2017-01-01', periods=7) + pd.Timedelta(days=self.weekly_start))
                df_w = plot.seasonality_plot_df(self.model, days)

                # Calculate seasonal components 
                self.forecast = self.model.predict_seasonal_components(df_w)

            elif self.seasonality == 'yearly':
                # Prepare the seasonality data frame
                # Parameter start needs to be 1st January for any arbitrary year
                days = (pd.date_range(start='2017-01-01', periods=365) + pd.Timedelta(days=self.yearly_start))
                df_y = plot.seasonality_plot_df(self.model, days)

                # Calculate seasonal components 
                self.forecast = self.model.predict_seasonal_components(df_y)

            else:
                # Prepare the seasonality data frame
                start = pd.to_datetime('2017-01-01 0000')
                period = self.model.seasonalities[self.seasonality]['period']
                
                end = start + pd.Timedelta(days=period)
                # plot_points = 200
                # plot_points is used instead of period below in fbprophet/forecaster.py. 
                # However, it seems to make more sense to use period given the expected usage in Qlik
                intervals = pd.to_datetime(np.linspace(start.value, end.value, period)) 
                
                df_x = plot.seasonality_plot_df(self.model, intervals)

                # Calculate seasonal components 
                self.forecast = self.model.predict_seasonal_components(df_x)
            
            # Set the correct sort order for the response
            try:
                self.forecast = self.forecast.reindex(self.sort_order.index)
            except AttributeError:
                pass
        
        # For standard forecast the output rows equal the input rows
        else:
            # Prepare the forecast
            self.forecast = self.model.predict(self.future_df)
            
            # For return=y_then_yhat[_upper / _lower] we return y values followed by relevant results for the forecast periods
            if 'y_then_yhat' in self.result_type:
                relevant_result = self.result_type.replace('y_then_', '')

                # Copy yhat / yhat_upper / yhat_lower values to the new column
                self.forecast.loc[:, self.result_type] = self.forecast.loc[:, relevant_result]

                if 'upper' in self.result_type or 'lower' in self.result_type:
                    # Overwrite historic values with Nulls
                    self.forecast.loc[:len(self.forecast) - self.periods - 1, self.result_type] \
                    = np.NaN
                else:
                    # Overwrite with y values for historic data
                    self.forecast.loc[:len(self.forecast) - self.periods - 1, self.result_type] \
                    = self.input_df.loc[:len(self.request_df) - self.periods - 1, 'y']
            
            # For return=residual we return y - yhat for historical periods and Null for future periods
            elif 'residual' in self.result_type:
                # Create the residuals for historical periods by subtracting yhat from y
                self.forecast.loc[:len(self.request_df)-self.periods-1, self.result_type] = self.input_df.loc[:len(self.request_df)-self.periods-1, 'y'] - self.forecast.loc[:len(self.request_df)-self.periods-1, 'yhat']
            
            # Update to the original index from the request data frame
            self.forecast.index = self.request_index.index
            
            # Reset to the original sort order of the data sent by Qlik
            self.forecast = self.forecast.sort_index()

        # Undo the logarithmic conversion if it was applied during initialization
        if self.take_log:
            if self.result_type == 'all':
                self.forecast.loc[:, self.forecast.columns != 'ds'] = np.exp(self.forecast.loc[:, self.forecast.columns != 'ds'])
            else:
                self.forecast.loc[:, self.result_type] = np.exp(self.forecast.loc[:, self.result_type])

        # Add back the null row if it was received in the request
        if len(self.NaT_df) > 0:
            if self.result_type == 'all':
                col = 'yhat'
            else:
                col = self.result_type
            self.NaT_df = self.NaT_df.rename({'y': col}, axis='columns')
            self.forecast = self.forecast.append(self.NaT_df)
        
    def _send_table_description(self):
        """
        Send the table description to Qlik as meta data.
        Only used when the SSE is called from the Qlik load script.
        """
        
        # Set up the table description to send as metadata to Qlik
        self.table = SSE.TableDescription()
        self.table.name = "ProphetForecast"
        self.table.numberOfRows = len(self.response)

        # Set up fields for the table
        if self.is_seasonality_request:
            for col in self.response.columns:
                self.table.fields.add(name=col, dataType=1)
        elif self.result_type == 'all':
            for col in self.response.columns:
                dataType = 0 if col == 'ds' else 1
                self.table.fields.add(name=col, dataType=dataType)
        else:
            self.table.fields.add(name="ds", dataType=0)
            self.table.fields.add(name=self.result_type, dataType=1)
        
        if self.debug:
            self._print_log(6)
        
        # Send table description
        table_header = (('qlik-tabledescription-bin', self.table.SerializeToString()),)
        self.context.send_initial_metadata(table_header)
    
    def _print_log(self, step):
        """
        Output useful information to stdout and the log file if debugging is required.
        step: Print the corresponding step in the log
        """
        
        # Set mode to append to log file
        mode = 'a'

        if step == 1:
            # Increment log counter for the class. Each instance of the class generates a new log.
            self.__class__.log_no += 1
             
            # Create a log file for the instance
            # Logs will be stored in ..\logs\Prophet Log <n>.txt
            self.logfile = os.path.join(os.getcwd(), 'logs', 'Prophet Log {}.txt'.format(self.log_no))
            
            # Output log header
            output = "ProphetForQlik Log: {0} \n\n".format(time.ctime(time.time()))
            # Set mode to write new log file
            mode = 'w'
        
        elif step == 2:
            # Output the request and input data frames
            output = "Prophet parameters: {0}\n\n".format(self.kwargs)
            output += "Instance creation parameters: {0}\n\n".format(self.prophet_kwargs)
            output += "Make future data frame parameters: {0}\n\n".format(self.make_kwargs)
            output += "Add seasonality parameters: {0}\n\n".format(self.add_seasonality_kwargs)
            output += "Fit parameters: {0}\n\n".format(self.fit_kwargs)
            if self.has_regressors and len(self.regressor_kwargs):
                output += "Additional regressor parameters: {0}\n\n".format(self.regressor_kwargs)
            output += "REQUEST DATA FRAME: {0} rows x cols\n\n".format(self.request_df.shape)
            output += "{0}\n...\n{1}\n\n".format(self.request_df.head(5).to_string(), self.request_df.tail(5).to_string())
            if len(self.NaT_df) > 0:
                output += "REQUEST NULL VALUES DATA FRAME: {0} rows x cols\n\n".format(self.NaT_df.shape)
                output += "{0} \n\n".format(self.NaT_df.to_string())
            output += "INPUT DATA FRAME: {0} rows x cols\n\n".format(self.input_df.shape)
            output += "{0}\n...\n{1}\n\n".format(self.input_df.head(5).to_string(), self.input_df.tail(5).to_string())
            if self.has_holidays:
                output += "HOLIDAYS DATA FRAME: {0} rows x cols\n\n".format(self.holidays_df.shape)
                output += "{0} \n\n".format(self.holidays_df.to_string())
        
        elif step == 3:
            # Output in case the input contains less than 2 non-Null rows
            output = "\nForecast cannot be generated as the request contains less than two non-Null rows\n\n"
        
        elif step == 4:
            # Output the future data frame 
            output = "\nFUTURE DATA FRAME: {0} rows x cols\n\n".format(self.future_df.shape)
            output += "{0}\n...\n{1}\n\n".format(self.future_df.head(5).to_string(), self.future_df.tail(5).to_string())

        elif step == 5:         
            # Output the forecast data frame and returned series 
            output = "\nFORECAST DATA FRAME: {0} rows x cols\n\n".format(self.forecast.shape)
            output += "RESULT COLUMNS:\n\n"
            for col in self.forecast:
                output += "{}\n".format(col)

            output += "\nSAMPLE RESULTS:\n{0} \n\n".format(self.forecast.tail(5).to_string())

            result = self.response if self.load_script else self.forecast
            cols = result.columns if self.result_type == 'all' else ['ds', self.result_type]
            output += "FORECAST RETURNED:\n{0}\n...\n{1}\n\n".format(result.loc[:, cols].head(5).to_string(),\
                result.loc[:, cols].tail(5).to_string())
        
        elif step == 6:
            # Print the table description if the call was made from the load script
            output = "\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table)

        elif step == 7:
            # Inform of fall back when additional regressors are incorrect
            output = "\nAdditional regressors have not been passed correctly. Falling back to a basic model.\n\n"
        
        sys.stdout.write(output)
        with open(self.logfile, mode, encoding='utf-8') as f:
            f.write(output)
    
    @staticmethod
    def timeit(request):
        """
        Time the different components of the forecast
        """
        
        import timeit
        import ServerSideExtension_pb2 as SSE
        
        # Create a log file for the 
        logfile = os.path.join(os.getcwd(), 'logs', 'Prophet Performance Log.txt')

        def t1(request):
            return ProphetForQlik(request)

        def t2(predictor):
            return predictor.predict()

        def t3(forecast):
            return forecast.apply(lambda result: iter([SSE.Dual(numData=result)]))

        def t4(response_rows):
            return response_rows.apply(lambda duals: SSE.Row(duals=duals)).tolist()
        
        def dotime1():
            t = timeit.Timer("t1(request)")
            time = t.timeit(1)
            sys.stdout.write("Time taken to create an instance of ProphetForQlik: {}\n".format(time))
            with open(logfile,'a') as f:
                f.write("Time taken to create an instance of ProphetForQlik: {}\n".format(time))
            
        predictor = ProphetForQlik(request)
        
        def dotime2():
            t = timeit.Timer("t2(predictor)")
            time = t.timeit(1)
            sys.stdout.write("Time taken to calculate the forecast: {}\n".format(time))
            with open(logfile,'a') as f:
                f.write("Time taken to calculate the forecast: {}\n".format(time))
        
        forecast = predictor.predict()
        
        def dotime3():
            t = timeit.Timer("t3(forecast)")
            time = t.timeit(1)
            sys.stdout.write("Time taken to convert results to SSE.Dual: {}\n".format(time))
            with open(logfile,'a') as f:
                f.write("Time taken to convert results to SSE.Dual: {}\n".format(time))
        
        response_rows = forecast.apply(lambda result: iter([SSE.Dual(numData=result)]))
        
        def dotime4():
            t = timeit.Timer("t4(response_rows)")
            time = t.timeit(1)
            sys.stdout.write("Time taken to convert duals to SSE.Row: {}\n".format(time))
            with open(logfile,'a') as f:
                f.write("Time taken to convert duals to SSE.Row: {}\n".format(time))

        import builtins
        builtins.__dict__.update(locals())

        dotime1()
        dotime2()
        dotime3()
        dotime4()