import os
import sys
import time
import string
import locale
import numpy as np
import pandas as pd
from fbprophet import Prophet, plot
import _utils as utils
import ServerSideExtension_pb2 as SSE

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
    
    def __init__(self, request):
        """
        Class initializer.
        :param request: an iterable sequence of RowData
        :Sets up the input data frame and parameters based on the request
        """
        
        # Set the request variable for this object instance
        self.request = request

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
        
        # If the request contains holidays create a holidays data frame
        if self.has_holidays:
            self.holidays_df = pd.DataFrame([(row.duals[0].numData, row.duals[2].strData)\
                                             for request_rows in self.request\
                                             for row in request_rows.rows],\
                                            columns=['ds','holiday'])
            
            if self.lower_window is not None:
                self.holidays_df.loc[:, 'lower_window'] = self.lower_window
                
            if self.upper_window is not None:
                self.holidays_df.loc[:, 'upper_window'] = self.upper_window
        
        # Additional information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(1)
        
        # Convert numerical date values to datetime
        self.request_df.loc[:,'ds'] = pd.to_datetime(self.request_df.loc[:,'ds'], unit=self.qlik_cal_unit,
                                                     origin=self.qlik_cal_start)
        
        # If the request contains holidays update the ds column for it as well
        if self.has_holidays:
            self.holidays_df.loc[:,'ds'] = self.request_df.loc[:,'ds'].copy()
            
            # Also remove rows from the holidays data frame where the holiday or ds column is empty
            self.holidays_df = self.holidays_df.loc[self.holidays_df.holiday != '']
            self.holidays_df = self.holidays_df.loc[self.holidays_df.ds.notnull()]
            
            # Make the holidays names lower case to avoid the return argument becoming case sensitive
            self.holidays_df.loc[:,'holiday'] = self.holidays_df.holiday.str.lower()
            # Also remove spaces and apostrophes
            self.holidays_df.loc[:,'holiday'] = self.holidays_df.holiday.str.replace(" ", "_")
            self.holidays_df.loc[:,'holiday'] = self.holidays_df.holiday.str.replace("'", "")
            
            # And sort by the ds column and reset indexes
            self.holidays_df = self.holidays_df.sort_values('ds')
            self.holidays_df = self.holidays_df.reset_index(drop=True)
            
            # Finally add this to the key word argumemnts for Prophet
            self.prophet_kwargs['holidays'] = self.holidays_df
        
        # Sort the Request Data Frame based on dates, as Qlik may send unordered data
        self.request_df = self.request_df.sort_values('ds')
        
        # Store the original indexes for re-ordering output later
        self.request_index = self.request_df.loc[:,'ds']
        
        # Ignore the placeholder rows which will be filled with forecasted figures later
        self.input_df = self.request_df.iloc[:-self.periods].copy()
        
        # Reset the indexes for the input data frame. 
        # Not doing this interferes with correct ordering of the output from Prophet
        self.input_df = self.input_df.reset_index(drop=True)
        
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
    def init_seasonality(cls, request):
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
        # The key word arguments are taken from the fourth column of the first row
        args = request[0].rows[0].duals[3]
        
        # The data may be sent unsorted by Qlik, so we have to store the order to use when sending the results
        sort_order = pd.DataFrame([(row.duals[0].numData, row.duals[0].strData) \
                                        for request_rows in request \
                                        for row in request_rows.rows], \
                                       columns=['seasonality_num', 'seasonality_str'])
        
        # We ignore Null values here as these are handled separately in the response
        sort_order = sort_order.loc[sort_order.seasonality_num.notnull()]
        
        # The correct sort order is based on the data frame's index after sorting on the seasonality field
        sort_order = sort_order.sort_values('seasonality_num')
        
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
            
            # Merge the holidays with the request data frame using column ds as key
            request_df = pd.merge(request_df, holiday_df, on='ds', how='left')
            
            # Replace null values in the holiday column with empty strings
            request_df = request_df.fillna(value={'holiday': ''})
        
        # Values in the data frame are converted to type SSE.Dual
        request_df.loc[:,'ds'] = request_df.loc[:,'ds'].apply(lambda result: SSE.Dual(numData=result))
        request_df.loc[:,'y'] = request_df.loc[:,'y'].apply(lambda result: SSE.Dual(numData=result))
        if 'holiday' in request_df.columns:
            request_df.loc[:,'holiday'] = request_df.loc[:,'holiday'].apply(lambda result: SSE.Dual(strData=result))
        
        # Add the keyword arguments to the data frame as well, already of type SSE.Dual
        request_df.loc[:, 'args'] = args
        
        # Create the updated request list and convert to SSE data types
        request_list = request_df.values.tolist()
        request_list = [SSE.Row(duals=duals) for duals in request_list]
        updated_request = [SSE.BundledRows(rows=request_list)]
                
        # Call the default initialization method
        instance = ProphetForQlik(updated_request)
        
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
        
        # Set the sort order to be used when returning the results
        instance.sort_order = sort_order
        
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
        
        self.model.fit(self.input_df)
             
        # Create a data frame for future values
        self.future_df = self.model.make_future_dataframe(**self.make_kwargs)
        
        # If a logistic growth model is applied add the cap and floor columns to the future data frame
        if self.cap is not None:
            self.future_df.loc[:,'cap'] = self.cap
            
            if self.floor is not None:
                self.future_df.loc[:,'floor'] = self.floor
        
        # Prepare the forecast
        self._forecast()
        
        if self.debug:
            self._print_log(4)
        
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
        self.is_seasonality_request = False
        self.weekly_start = 6 # Defaulting to a Monday start for the week as used in Qlik
        self.yearly_start = 0
        self.lower_window = None
        self.upper_window = None
        
        # Set optional parameters
        
        # Check if there is a fourth column in the request
        try:
            # If there is a fourth column, it is assumed to contain the key word arguments
            args = self.request[0].rows[0].duals[3].strData
            
            # The third column should then provide the holiday name or null for each row
            self.has_holidays = True
            
        except IndexError:
            # If there is no fourth column, the request does not include holidays
            self.has_holidays = False
        
        # If the fourth column did not exist, we try again with the third column
        if not self.has_holidays:
            try:
                args = self.request[0].rows[0].duals[2].strData
            except IndexError:
                args = None
        
        # If the key word arguments were included in the request, get the parameters and values
        if args is not None:
            
            # The parameter and values are transformed into key value pairs
            args = args.translate(str.maketrans('', '', string.whitespace)).split(",")
            self.kwargs = dict([arg.split("=") for arg in args])
            
            # Make sure the key words are in lower case
            self.kwargs = {k.lower(): v for k, v in self.kwargs.items()}
            
            # Set the return type 
            # Valid values are: yhat, trend, seasonal, seasonalities. 
            # Add _lower or _upper to the series name to get lower or upper limits.
            if 'return' in self.kwargs:
                self.result_type = self.kwargs['return'].lower()
            
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
        
        # Create dictionary of arguments for the Prophet(), make_future_dataframe() and add_seasonality() functions
        self.prophet_kwargs = {}
        self.make_kwargs = {}
        self.add_seasonality_kwargs = {}
        
        # Populate the parameters in the corresponding dictionary:
        
        # Set up a list of possible key word arguments for the Prophet() function
        prophet_params = ['seasonality_mode', 'growth', 'changepoint_prior_scale', 'interval_width',\
                          'seasonality_prior_scale', 'holidays_prior_scale']
        
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
            self.forecast = self.forecast.reindex(self.sort_order.index)
        
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
            
            # Update to the original index from the request data frame
            self.forecast.index = self.request_index.index
            
            # Reset to the original sort order of the data sent by Qlik
            self.forecast = self.forecast.sort_index()

        # Undo the logarithmic conversion if it was applied during initialization
        if self.take_log:
            self.forecast.loc[:,self.result_type] = np.exp(self.forecast.loc[:,self.result_type])

        # Add back the null row if it was received in the request
        if len(self.NaT_df) > 0:
            self.NaT_df = self.NaT_df.rename({'y': self.result_type}, axis='columns')
            self.forecast = self.forecast.append(self.NaT_df)
        
    def _print_log(self, step):
        """
        Output useful information to stdout and the log file if debugging is required.
        step: Print the corresponding step in the log
        """
        
        if step == 1:
            # Increment log counter for the class. Each instance of the class generates a new log.
            self.__class__.log_no += 1
             
            # Create a log file for the instance
            # Logs will be stored in ..\logs\Prophet Log <n>.txt
            self.logfile = os.path.join(os.getcwd(), 'logs', 'Prophet Log {}.txt'.format(self.log_no))
            
            # Output log header
            sys.stdout.write("ProphetForQlik Log: {0} \n\n".format(time.ctime(time.time())))
            with open(self.logfile,'w') as f:
                f.write("ProphetForQlik Log: {0} \n\n".format(time.ctime(time.time())))
        
        elif step == 2:
            # Output the request and input data frames to the terminal
            sys.stdout.write("Prophet parameters: {0}\n\n".format(self.kwargs))
            sys.stdout.write("Instance creation parameters: {0}\n\n".format(self.prophet_kwargs))
            sys.stdout.write("Make future data frame parameters: {0}\n\n".format(self.make_kwargs))
            sys.stdout.write("Add seasonality parameters: {0}\n\n".format(self.add_seasonality_kwargs))
            sys.stdout.write("REQUEST DATA FRAME: {0} rows x cols\n\n".format(self.request_df.shape))
            sys.stdout.write("{0} \n\n".format(self.request_df.to_string()))
            if len(self.NaT_df) > 0:
                sys.stdout.write("REQUEST NULL VALUES DATA FRAME: {0} rows x cols\n\n".format(self.NaT_df.shape))
                sys.stdout.write("{0} \n\n".format(self.NaT_df.to_string()))
            sys.stdout.write("INPUT DATA FRAME: {0} rows x cols\n\n".format(self.input_df.shape))
            sys.stdout.write("{} \n\n".format(self.input_df.to_string()))
            if self.has_holidays:
                sys.stdout.write("HOLIDAYS DATA FRAME: {0} rows x cols\n\n".format(self.holidays_df.shape))
                sys.stdout.write("{0} \n\n".format(self.holidays_df.to_string()))
            
            # Output the request and input data frames to the log file 
            with open(self.logfile,'a') as f:
                f.write("Prophet parameters: {0}\n\n".format(self.kwargs))
                f.write("Instance creation parameters: {0}\n\n".format(self.prophet_kwargs))
                f.write("Make future data frame parameters: {0}\n\n".format(self.make_kwargs))
                f.write("Add seasonality parameters: {0}\n\n".format(self.add_seasonality_kwargs))
                f.write("REQUEST DATA FRAME: {0} rows x cols\n\n".format(self.request_df.shape))
                f.write("{0} \n\n".format(self.request_df.to_string()))
                if len(self.NaT_df) > 0:
                    f.write("REQUEST NULL VALUES DATA FRAME: {0} rows x cols\n\n".format(self.NaT_df.shape))
                    f.write("{0} \n\n".format(self.NaT_df.to_string()))
                f.write("INPUT DATA FRAME: {0} rows x cols\n\n".format(self.input_df.shape))
                f.write("{0} \n\n".format(self.input_df.to_string()))
                if self.has_holidays:
                    f.write("HOLIDAYS DATA FRAME: {0} rows x cols\n\n".format(self.holidays_df.shape))
                    f.write("{0} \n\n".format(self.holidays_df.to_string()))
        
        elif step == 3:
            # Output in case the input contains less than 2 non-Null rows
            sys.stdout.write("\nForecast cannot be generated as the request contains less than two non-Null rows\n\n")
            with open(self.logfile,'a') as f:
                f.write("\nForecast cannot be generated as the request contains less than two non-Null rows\n\n")
        
        elif step == 4:         
            # Output the forecast data frame and returned series to the terminal
            sys.stdout.write("\nFORECAST DATA FRAME: {0} rows x cols\n\n".format(self.forecast.shape))
            sys.stdout.write("RESULT COLUMNS:\n\n")
            [sys.stdout.write("{}\n".format(col)) for col in self.forecast]
            sys.stdout.write("\nSAMPLE RESULTS:\n{0} \n\n".format(self.forecast.tail(self.periods).to_string()))
            sys.stdout.write("FORECAST RETURNED:\n{0}\n\n".format(self.forecast.loc[:,self.result_type].to_string()))
            
            # Output the forecast data frame and returned series to the log file
            with open(self.logfile,'a') as f:
                f.write("\nFORECAST DATA FRAME: {0} rows x cols\n\n".format(self.forecast.shape))
                f.write("RESULT COLUMNS:\n\n")
                [f.write("{}\n".format(col)) for col in self.forecast]
                f.write("\nSAMPLE RESULTS:\n{0} \n\n".format(self.forecast.tail(self.periods).to_string()))
                f.write("FORECAST RETURNED:\n{0}\n\n".format(self.forecast.loc[:,self.result_type].to_string()))
    
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