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

import spacy
import _utils as utils
import ServerSideExtension_pb2 as SSE

# Add Generated folder to module path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

class SpaCyForQlik:
    """
    A class to implement spaCy natural language processing capabilities for Qlik.
    https://spacy.io/
    """
    
    # Counter used to name log files for instances of the class
    log_no = 0

    def __init__(self, request, context, path="../models/spaCy/"):
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
    
    def get_entities(self, default=True):
        """
        Use spaCy NER to return named entities from text.
        :
        :default=True uses the pre-trained English language models provided by spaCy. 
        :default=False allows the use of a re-trained spaCy model.
        """

        if default:
            # Interpret the request data based on the expected row and column structure
            row_template = ['strData', 'strData', 'strData']
            col_headers = ['key', 'text', 'kwargs']
        else:
            # A model name is required if using a custom spaCy model
            row_template = ['strData', 'strData', 'strData', 'strData']
            col_headers = ['key', 'text', 'model_name', 'kwargs']
        
        # Create a Pandas DataFrame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)

        # Get the argument strings from the request dataframe
        kwargs = self.request_df.loc[0, 'kwargs']
        # Set the relevant parameters using the argument strings
        self._set_params(kwargs)

        # Print the request dataframe to the logs
        if self.debug:
            self._print_log(3)

        # Extract named entities for each text in the request dataframe
        self.response_df = self._entity_tagger()

        # Print the response dataframe to the logs
        if self.debug:
            self._print_log(4)

        # Send the reponse table description to Qlik
        self._send_table_description("entities")
        
        return self.response_df

    def _set_params(self, kwargs):
        """
        Set input parameters based on the request.
        :
        :Parameters used by this SSE are: 
        :overwrite, debug
        :For details refer to the GitHub project: https://github.com/nabeel-oz/qlik-py-tools
        """
        
        # Set default values which will be used if execution arguments are not passed
        
        # Default parameters:
        self.overwrite = False
        self.debug = False
        self.model = 'en_core_web_sm'
              
        # Extract the model path if required
        try:
            # Get the model name from the first row in the request_df and prefix it with the path
            self.model = self.path + self.request_df.loc[0, ['model_name']]
            # Remove the model_name column from the request_df
            self.request_df = self.request_df.drop(['model_name'], axis=1)
        except KeyError:
            pass
        
        # If key word arguments were included in the request, get the parameters and values
        if len(kwargs) > 0:
            
            # Transform the string of arguments into a dictionary
            self.kwargs = utils.get_kwargs(kwargs)
            
            # Set the overwite parameter if any existing model with the specified name should be overwritten
            if 'overwrite' in self.kwargs:
                self.overwrite = 'true' == self.kwargs['overwrite'].lower()
            
            # Set the debug option for generating execution logs
            # Valid values are: true, false
            if 'debug' in self.kwargs:
                self.debug = 'true' == self.kwargs['debug'].lower()
                
                # Additional information is printed to the terminal and logs if the paramater debug = true
                if self.debug:
                    # Increment log counter for the class. Each instance of the class generates a new log.
                    self.__class__.log_no += 1

                    # Create a log file for the instance
                    # Logs will be stored in ..\logs\SpaCy Log <n>.txt
                    self.logfile = os.path.join(os.getcwd(), 'logs', 'SpaCy Log {}.txt'.format(self.log_no))

                    self._print_log(1)
                
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(2)
        
        # Remove the kwargs column from the request_df
        self.request_df = self.request_df.drop(['kwargs'], axis=1)

    def _entity_tagger(self):
        """
        Get named entities from the spaCy model for each text in the request dataframe.
        """

        # Load the spaCy model
        nlp = spacy.load(self.model)
        
        # Create an empty list for storing named entities
        entities = []

        # Send each text to the model and save the named entities
        for i in range(len(self.request_df)):
            key = self.request_df.loc[i, 'key']
            doc = nlp(self.request_df.loc[i, 'text'])

            # Obtain entities, start and end characters, labels and descriptions    
            for ent in doc.ents:
                entities.append([key, ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_)])

        # Transform the entities list to a dataframe
        entities = pd.DataFrame(entities, columns=['key', 'entity', 'start', 'end', 'type', 'description'])
        
        return entities

    def _send_table_description(self, variant):
        """
        Send the table description to Qlik as meta data.
        Used when the SSE is called from the Qlik load script.
        """
        
        # Set up the table description to send as metadata to Qlik
        self.table = SSE.TableDescription()
        self.table.name = "SSE-Response-spaCy"
        self.table.numberOfRows = len(self.response_df)

        # Set up fields for the table
        if variant == "entities":
            self.table.fields.add(name="key")
            self.table.fields.add(name="entity")
            self.table.fields.add(name="start", dataType=1)
            self.table.fields.add(name="end", dataType=1)
            self.table.fields.add(name="type")
            self.table.fields.add(name="description")
                
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
        
        if self.logfile is None:
            # Increment log counter for the class. Each instance of the class generates a new log.
            self.__class__.log_no += 1

            # Create a log file for the instance
            # Logs will be stored in ..\logs\SKLearn Log <n>.txt
            self.logfile = os.path.join(os.getcwd(), 'logs', 'SpaCy Log {}.txt'.format(self.log_no))
        
        if step == 1:
            # Output log header
            sys.stdout.write("\nSpaCyForQlik Log: {0} \n\n".format(time.ctime(time.time())))
            
            with open(self.logfile,'w', encoding='utf-8') as f:
                f.write("SpaCyForQlik Log: {0} \n\n".format(time.ctime(time.time())))
                                
        elif step == 2:
            # Output the model name to the terminal
            sys.stdout.write("Model: {0}\n\n".format(self.model))
            # Output the execution parameters to the terminal
            sys.stdout.write("Execution parameters: {0}\n\n".format(self.kwargs))
            
            # Output the same information to the log file 
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("Model: {0}\n\n".format(self.model))
                f.write("Execution parameters: {0}\n\n".format(self.kwargs))
        
        elif step == 3:
            # Output the request data frame to the terminal
            sys.stdout.write("REQUEST: {0} rows x cols\nSample Data:\n\n".format(self.request_df.shape))
            sys.stdout.write("{0}\n...\n{1}\n\n".format(self.request_df.head().to_string(), self.request_df.tail().to_string()))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("REQUEST: {0} rows x cols\nSample Data:\n\n".format(self.request_df.shape))
                f.write("{0}\n...\n{1}\n\n".format(self.request_df.head().to_string(), self.request_df.tail().to_string()))
        
        elif step == 4:
            # Output the response data frame to the terminal
            sys.stdout.write("RESPONSE: {0} rows x cols\nSample Data:\n\n".format(self.response_df.shape))
            sys.stdout.write("{0}\n...\n{1}\n\n".format(self.response_df.head().to_string(), self.response_df.tail().to_string()))
            
            with open(self.logfile,'a', encoding='utf-8') as f:
                f.write("RESPONSE: {0} rows x cols\nSample Data:\n\n".format(self.response_df.shape))
                f.write("{0}\n...\n{1}\n\n".format(self.response_df.head().to_string(), self.response_df.tail().to_string()))
        
        elif step == 5:
            # Print the table description if the call was made from the load script
            sys.stdout.write("\nTABLE DESCRIPTION SENT TO QLIK:\n\n{0} \n\n".format(self.table))
            
            # Write the table description to the log file
            with open(self.logfile,'a', encoding='utf-8') as f:
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