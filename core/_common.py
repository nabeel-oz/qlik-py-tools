import os
import sys
import time
import string
import pathlib
import warnings
import numpy as np
import pandas as pd

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from efficient_apriori import apriori
import _utils as utils
import ServerSideExtension_pb2 as SSE

# Add Generated folder to module path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

class CommonFunction:
    """
    A class to implement common data science functions for Qlik.
    """
    
    # Counter used to name log files for instances of the class
    log_no = 0

    def __init__(self, request, context):
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
        self.logfile = None
    
    def association_rules(self):
        """
        Use an apriori algorithm to uncover association rules between items.
        """

        # Interpret the request data based on the expected row and column structure
        self._initiate(row_template = ['strData', 'strData', 'strData'],  col_headers = ['group', 'item', 'kwargs'])

        # Create a list of items for each group
        transactions = []

        # Iterate over each group and add a tuple of items to the list
        for group in self.request_df['group'].unique():
            transactions.append(tuple(self.request_df.item[self.request_df.group == group]))

        # Get the item sets and association rules from the apriori algorithm
        itemsets, rules = apriori(transactions, **self.pass_on_kwargs)

        # Prepare the response
        response = []

        # for each rule get the left hand side and right hand side together with support, confidence and lift 
        for rule in sorted(rules, key=lambda rule: rule.lift, reverse=True):
            lhs = ", ".join(map(str, rule.lhs))
            rhs = ", ".join(map(str, rule.rhs))
            desc = "{0} -> {1}".format(lhs, rhs)
            response.append((desc, lhs, rhs, rule.support, rule.confidence, rule.lift))

        # if no association rules were found the parameters may need to be adjusted
        if len(response) == 0:
            err = "No association rules could be found. You may get results by lowering the limits imposed by " + \
                "the min_support and min_confidence parameters.\ne.g. by passing min_support=0.2|float in the arguments."
            raise Exception(err) 

        self.response_df = pd.DataFrame(response, columns=['rule', 'rule_lhs', 'rule_rhs', 'support', 'confidence', 'lift'])

        # Print the response dataframe to the logs
        if self.debug:
            self._print_log(4)

        # Send the reponse table description to Qlik
        self._send_table_description("apriori")
        
        return self.response_df
    
    def _initiate(self, row_template, col_headers):
        """
        Interpret the request data and setup execution parameters
        :
        :row_template: a list of data types expected in the request e.g. ['strData', 'numData']
        :col_headers: a list of column headers for interpreting the request data e.g. ['group', 'item']
        """
                
        # Create a Pandas DataFrame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)

        # Get the argument strings from the request dataframe
        kwargs = self.request_df.loc[0, 'kwargs']
        # Set the relevant parameters using the argument strings
        self._set_params(kwargs)

        # Print the request dataframe to the logs
        if self.debug:
            self._print_log(3)
    
    def _set_params(self, kwargs):
        """
        Set input parameters based on the request.
        :
        :For details refer to the GitHub project: https://github.com/nabeel-oz/qlik-py-tools
        """
        
        # Set default values which will be used if execution arguments are not passed
        
        # Default parameters:
        self.debug = False
        
        # If key word arguments were included in the request, get the parameters and values
        if len(kwargs) > 0:
            
            # Transform the string of arguments into a dictionary
            self.kwargs = utils.get_kwargs(kwargs)
            
            # Set the debug option for generating execution logs
            # Valid values are: true, false
            if 'debug' in self.kwargs:
                self.debug = 'true' == self.kwargs.pop('debug').lower()
                
                # Additional information is printed to the terminal and logs if the paramater debug = true
                if self.debug:
                    # Increment log counter for the class. Each instance of the class generates a new log.
                    self.__class__.log_no += 1

                    # Create a log file for the instance
                    # Logs will be stored in ..\logs\Common Functions Log <n>.txt
                    self.logfile = os.path.join(os.getcwd(), 'logs', 'Common Functions Log {}.txt'.format(self.log_no))

                    self._print_log(1)
            
            # Get the rest of the parameters, converting values to the correct data type
            self.pass_on_kwargs = utils.get_kwargs_by_type(self.kwargs) 
                          
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(2)
        
        # Remove the kwargs column from the request_df
        self.request_df = self.request_df.drop(['kwargs'], axis=1)

    def _send_table_description(self, variant):
        """
        Send the table description to Qlik as meta data.
        Used when the SSE is called from the Qlik load script.
        """
        
        # Set up the table description to send as metadata to Qlik
        self.table = SSE.TableDescription()
        self.table.name = "SSE-Response"
        self.table.numberOfRows = len(self.response_df)

        # Set up fields for the table
        if variant == "apriori":
            'rule', 'rule_lhs', 'rule_rhs', 'support', 'confidence', 'lift'
            self.table.fields.add(name="rule")
            self.table.fields.add(name="rule_lhs")
            self.table.fields.add(name="rule_rhs")
            self.table.fields.add(name="support", dataType=1)
            self.table.fields.add(name="confidence", dataType=1)
            self.table.fields.add(name="lift", dataType=1)
                
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(5)
            
        # Send table description
        table_header = (('qlik-tabledescription-bin', self.table.SerializeToString()),)
        self.context.send_initial_metadata(table_header)
    
    def _print_log(self, step):
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
            self.logfile = os.path.join(os.getcwd(), 'logs', 'Common Functions Log {}.txt'.format(self.log_no))
        
        if step == 1:
            # Output log header
            output = "\nCommonFunction Log: {0} \n\n".format(time.ctime(time.time()))
            # Set mode to write new log file
            mode = 'w'
                                
        elif step == 2:
            # Output the execution parameters to the terminal and log file
            output = "Execution parameters: {0}\n\n".format(self.kwargs) 
        
        elif step == 3:
            # Output the request data frame to the terminal and log file
            output = "REQUEST: {0} rows x cols\nSample Data:\n\n".format(self.request_df.shape)
            output += "{0}\n...\n{1}\n\n".format(self.request_df.head().to_string(), self.request_df.tail().to_string())
        
        elif step == 4:
            # Output the response data frame to the terminal and log file
            output = "RESPONSE: {0} rows x cols\nSample Data:\n\n".format(self.response_df.shape)
            output += "{0}\n...\n{1}\n\n".format(self.response_df.head().to_string(), self.response_df.tail().to_string())
        
        elif step == 5:
            # Output the table description if the call was made from the load script
            output = "\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table)

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
        
        if self.debug:
            with open(self.logfile,'a') as f:
                f.write("\n{0}: {1} \n\n".format(s, e))