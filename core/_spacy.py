import os
import re
import sys
import time
import string
import pathlib
import joblib
import random
import warnings
import numpy as np
import pandas as pd
from copy import copy

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import spacy
from spacy.util import minibatch, compounding, decaying
from spacy.gold import GoldParse
from sklearn.model_selection import train_test_split
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
    
    def retrain(self):
        """
        Retrain a spacy model for NER using training data.
        """

        # The request provides training data texts, entities, entity types together with the model name and any other arguments
        row_template = ['strData', 'strData', 'strData', 'strData', 'strData']
        col_headers = ['text', 'entity', 'entity_type', 'model_name', 'kwargs']
        
        # Create a Pandas DataFrame for the request data
        self.request_df = utils.request_df(self.request, row_template, col_headers)

        # Get the argument strings from the request dataframe
        kwargs = self.request_df.loc[0, 'kwargs']
        # Set the relevant parameters using the argument strings
        self._set_params(kwargs)

        # Check that a model name has been set
        if self.model in ["en_core_web_sm"]:
            err = "Incorrect usage: A name for the custom model needs to be specified."
            raise Exception(err)
        
        # Transform the training data to spaCy's training data format
        # This call populates the self.train and self.validation (if a test set is specified in the request arguments) objects
        self._prep_data()

        # Retrain the model and calculate evaluation metrics
        # This call saves the retrained model to disk and pepares the self.metrics dataframe for the response
        self._retrain_model()

        # Prepare the response, which will be the evaluation metrics prepared during retraining
        self.response_df = self.metrics

        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(11)
        
        # Send the reponse table description to Qlik
        self._send_table_description("metrics")
        
        # Finally send the response
        return self.response_df
    
    def _set_params(self, kwargs):
        """
        Set input parameters based on the request.
        :
        :For details refer to the GitHub project: https://github.com/nabeel-oz/qlik-py-tools
        """
        
        # Set default values which will be used if execution arguments are not passed
        
        # Default parameters:
        self.debug = False
        self.model = 'en_core_web_sm'
        self.custom = False
        self.base_model = 'en_core_web_sm'
        self.blank = False
        self.epochs = 100
        self.batch_size = compounding(4.0, 32.0, 1.001)
        self.drop = 0.25
        self.test = 0
              
        # Extract the model path if required
        try:
            # Get the model name from the first row in the request_df 
            self.model = self.request_df.loc[0, 'model_name']

            # Remove the model_name column from the request_df
            self.request_df = self.request_df.drop(['model_name'], axis=1)
        except KeyError:
            pass
        
        # If key word arguments were included in the request, get the parameters and values
        if len(kwargs) > 0:
            
            # Transform the string of arguments into a dictionary
            self.kwargs = utils.get_kwargs(kwargs)
            
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
            
            # Set whether the model (if getting named entites) or base model (if retraining) is a custom model
            # i.e. not one of the pre-trained models provided by spaCy
            if 'custom' in self.kwargs:
                self.custom = 'true' == self.kwargs['custom'].lower()
           
            # Set the base model, i.e an existing spaCy model to be retrained.
            if 'base_model' in self.kwargs:
                self.base_model = self.kwargs['base_model'].lower()
            
            # Set the retraining to be done on a blank Language class
            if 'blank' in self.kwargs:
                self.blank = 'true' == self.kwargs['blank'].lower()
            
            # Set the epochs for training the model. 
            # This is the the number times that the learning algorithm will work through the entire training dataset.
            # Valid values are an integer e.g. 200
            if 'epochs' in self.kwargs:
                self.epochs = utils.atoi(self.kwargs['epochs'])
            
            # Set the batch size to be used during model training. 
            # The model's internal parameters will be updated at the end of each batch.
            # Valid values are a single integer or compounding or decaying parameters.
            if 'batch_size' in self.kwargs:
                # The batch size may be a single integer
                try:
                    self.batch_size = utils.atoi(self.kwargs['batch_size'])
                # Or a list of floats
                except ValueError:
                    sizes = utils.get_kwargs_by_type(self.kwargs['batch_size']) 

                    # If the start < end, batch sizes will be compounded
                    if sizes[0] < sizes[1]:
                        self.batch_size = compounding(sizes[0], sizes[1], sizes[2])
                    # else bath sizes will decay during training
                    else:
                        self.batch_size = decaying(sizes[0], sizes[1], sizes[2])
            
            # Set the dropout rate for retraining the model
            # This determines the likelihood that a feature or internal representation in the model will be dropped,
            # making it harder for the model to memorize the training data.
            # Valid values are a float lesser than 1.0 e.g. 0.35
            if 'drop' in self.kwargs:
                self.drop = utils.atof(self.kwargs['drop'])
            
            # Set the ratio of data to be used for testing. 
            # This data will be held out from training and just used to provide evaluation metrics.
            # Valid values are a float >= zero and < 1.0 e.g. 0.3
            if 'test' in self.kwargs:
                self.test = utils.atof(self.kwargs['test'])
               
        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(2)
        
        # Remove the kwargs column from the request_df
        self.request_df = self.request_df.drop(['kwargs'], axis=1)

    def _entity_tagger(self):
        """
        Get named entities from the spaCy model for each text in the request dataframe.
        """

        # If this is a custom model, set the path to the directory
        if self.custom:
            self.model = self.path + self.model + "/"

        # Load the spaCy model
        try:
            nlp = spacy.load(self.model)
        except OSError:
            self.model = self.path + self.model + "/"
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

    def _prep_data(self):
        """
        Prepare the data for retraining the model.
        This transforms the data into spaCy's training data format with tuples of text and entity offsets.
        """

        # Firstly, we transform the dataframe which has repeated texts with one row per entity...
        # to a list of each text with its corresponding dictionary of entities.

        prev_text = ''
        self.train = []
        entities = {"entities": []}

        # For each sample in the dataframe
        for i in self.request_df.index:
            # Extract the text for the current index
            text = self.request_df.loc[i, 'text']    

            # If this is not the first record and we have reached a new text
            if i > 0 and text != prev_text:
                # Add the text and dictionary of entities to the training set
                self.train.append((prev_text, entities))

                # Reset variables
                entities = {"entities": []}
                prev_text = text
            # For the first record we set previous text to this text
            elif i == 0:
                prev_text = text

            # Extract the entity and entity type for the current index
            entity = (self.request_df.loc[i, 'entity'], self.request_df.loc[i, 'entity_type'])

            # Add entity to the entities dictionary 
            entities["entities"].append(entity)

        # Add the final text and dictionary of entities to the training set
        self.train.append((prev_text, entities))
        
        # Print the semi-transformed data to the logs
        if self.debug:
            self._print_log(6)

        # Complete the data prep by calculating entity offsets and finalizing the format for spaCy

        # Format the training data for spaCy
        for sample in self.train:
            # Get the text and named entities for the current sample
            text = sample[0]
            entities = sample[1]["entities"]
            entity_boundaries = []
            
            # Structure the entities and types into a DataFrame
            entities_df = pd.DataFrame(zip(*entities)).T
            entities_df.columns = ['ents', 'types']
            
            # For each unique entity
            for entity in entities_df.ents.unique():
                
                # Set up a regex pattern to look for the entity w.r.t. word boundaries 
                pattern = re.compile(r"\b" + entity + r"\b")

                # Get entity types for the entity. This may be a series of values if the entity appears more than once.
                types = entities_df[entities_df.ents == entity].types.reset_index(drop=True)
                has_multiple_types = True if len(types.unique()) > 1 else False
                i = 0
                
                # Find all occurrences of the entity in the text
                for match in re.finditer(pattern, text):
                    entity_boundaries.append((match.start(), match.end(), types[i]))

                    # Assign types according to the series
                    if has_multiple_types:
                        i += 1
                        
            if len(entity_boundaries)  > 0:
                # Prepare variables to check for overlapping entity boundaries
                start, stop, entity_type = map(list, zip(*entity_boundaries))

                # Drop overlapping entities, i.e. where an entity is a subset of a longer entity
                for i in range(len(start)):
                    other_start, other_stop = copy(start), copy(stop)
                    del other_start[i]
                    del other_stop[i]
                    
                    for j in range(len(other_start)):
                        if start[i] >= other_start[j] and stop[i] <= other_stop[j]:
                            entity_boundaries.remove((start[i], stop[i], entity_type[i]))
            
            # Add the entity boundaries to the sample
            sample[1]["entities"] = entity_boundaries

        # If required, split the data into training and testing sets
        if self.test > 0:
            self.train, self.validation = train_test_split(self.train, test_size=self.test)
        # Otherwise use the entire dataset for training
        else:
            self.validation = None

        # Print the final training data to the logs
        if self.debug:
            self._print_log(6)
    
    def _retrain_model(self, locked_timeout=2):
        """
        Update an existing spaCy model with labelled training data.
        The model is stored to disk using spaCy's to_disk method.
        If the model is found to be locked (based on the existance of a lock file) this function will fail after 'locked_timeout' seconds.
        """

        # Load the model, set up the pipeline and train the entity recognizer:
        
        # Load existing spaCy model
        if not self.blank:
            # If this is a custom model, set the path to the directory
            if self.custom:
                self.base_model = self.path + self.base_model + "/"
            
            nlp = spacy.load(self.base_model)  
        # If the parameter blank=true is passed we start with a blank Language class, e.g. en
        else:
            nlp = spacy.blank(self.base_model)  

        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(7)

        # create the built-in pipeline components and add them to the pipeline

        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
        else:
            ner = nlp.get_pipe("ner")

        # add labels
        for _, annotations in self.train:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Retrain the model:
        
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):  # only train NER
           
            # Setup lists to store the loss for each epoch
            self.losses_train = []
            self.losses_test = []
            
            # reset and initialize the weights randomly – but only if we're
            # training a new model
            if self.blank:
                nlp.begin_training()
            for epoch in range(self.epochs): 
                random.shuffle(self.train)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(self.train, size=self.batch_size)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=self.drop,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
                # Store loss for the epoch to a list
                self.losses_train.append(('Epoch {}'.format(epoch+1), losses['ner']))

                # Debug information is printed to the terminal and logs if the paramater debug = true
                if self.debug:
                    self._print_log(8)
                
                # If a test dataset is available, calculate losses for it as well
                if self.validation is not None:
                    losses = {}

                    # batch up the examples using spaCy's minibatch
                    batches = minibatch(self.validation, size=self.batch_size)
                    for batch in batches:
                        texts, annotations = zip(*batch)
                        # Get losses for the test data without updating the model 
                        nlp.update(
                            texts,  # batch of texts
                            annotations,  # batch of annotations
                            sgd = None,  # do not update model weights
                            losses=losses,
                        )
                    # Store loss for the epoch to a list
                    self.losses_test.append(('Epoch {}'.format(epoch+1), losses['ner']))

                    # Debug information is printed to the terminal and logs if the paramater debug = true
                    if self.debug:
                        self._print_log(9)

        # Save model to output directory:
        
        output_dir = pathlib.Path(self.path + self.model + '/')
        lock_file = pathlib.Path(self.path + self.model + '.lock')
        
        # Create the output directory if required
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)
        # If the model is locked wait a few seconds
        elif lock_file.exists():
            time.sleep(locked_timeout)
            # If the model is still locked raise an exception
            if lock_file.exists():
                raise TimeoutError("The specified model is locked. If you think this is wrong, please delete file {0}".format(lock_file))

        try:
            # Create the lock file
            joblib.dump(self.path + self.model, filename=lock_file)

            # Store the spaCy model to disk
            nlp.to_disk(output_dir)
        finally:
            # Delete the lock file
            lock_file.unlink()

        # Debug information is printed to the terminal and logs if the paramater debug = true
        if self.debug:
            self._print_log(10)

        # Evaluate the model:

        # Prepare spaCy docs and golds for getting evaluation metrics
        docs_golds = []
        for sample in self.train:
            doc = nlp.make_doc(sample[0])
            gold = GoldParse(doc, entities=sample[1]["entities"])
            docs_golds.append((doc, gold))
        
        # Get scores for training data
        scorer_train = nlp.evaluate(docs_golds)
        # Add the training scores to evaluation metrics
        self.metrics = self._prep_scores(scorer_train)

        # Get scores for testing data and add to the evaluation metrics
        if self.validation is not None:
            docs_golds = []
            for sample in self.validation:
                doc = nlp.make_doc(sample[0])
                gold = GoldParse(doc, entities=sample[1]["entities"])
                docs_golds.append((doc, gold))
            
            scorer_test = nlp.evaluate(docs_golds)
            self.metrics = pd.concat([self.metrics, self._prep_scores(scorer_test, subset='test')], ignore_index=True)

        # Add loss metrics
        self.metrics = pd.concat([self.metrics, self._prep_losses(self.losses_train)], ignore_index=True)
        if self.validation is not None:
            self.metrics = pd.concat([self.metrics, self._prep_losses(self.losses_test, subset='test')], ignore_index=True)
    
    def _prep_scores(self, scorer, subset='train'):
        """
        Prepare score metrics using a spaCy scrorer 
        Returns a dataframe formatted for output
        """
        columns = ["metric", "value"]

        # Prepare scorer metrics
        scores = {"Precision": scorer.scores["ents_p"], "Recall": scorer.scores["ents_r"], "F-score": scorer.scores["ents_f"]}
        metrics_df = pd.DataFrame([(k, v) for k, v in scores.items()], columns=columns)

        metrics_df.loc[:,'model'] = self.model
        metrics_df.loc[:,'subset'] = subset
        metrics_df = metrics_df[['model', 'subset'] + columns]

        return metrics_df
    
    def _prep_losses(self, losses, subset='train'):
        """
        Prepare loss metrics using a list of tuples of the format: (epoch, loss)
        Where epoch is an integer and loss is a float 
        Returns a dataframe formatted for output
        """
        columns = ["metric", "value"]
        metrics_df = pd.DataFrame(losses, columns=columns)
        metrics_df.loc[:,'model'] = self.model
        metrics_df.loc[:,'subset'] = subset
        metrics_df = metrics_df[['model', 'subset'] + columns]

        return metrics_df

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
        elif variant == "metrics":
            self.table.fields.add(name="model_name")
            self.table.fields.add(name="subset")
            self.table.fields.add(name="metric")
            self.table.fields.add(name="value", dataType=1)
                
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
            self.logfile = os.path.join(os.getcwd(), 'logs', 'SpaCy Log {}.txt'.format(self.log_no))
        
        if step == 1:
            # Output log header
            output = "\nSpaCyForQlik Log: {0} \n\n".format(time.ctime(time.time()))
            # Set mode to write new log file
            mode = 'w'
                                
        elif step == 2:
            # Output the model name and execution parameters to the terminal and log file
            output = "Model: {0}\n\n".format(self.model)
            output += "Execution parameters: {0}\n\n".format(self.kwargs) 
        
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
        
        elif step == 6:
            output = "Completed transformations.\nTop and bottom 3 samples for training:\n\n"
            
            # Output the top and bottom 3 results from training data
            for text, annotations in self.train[:3] + self.train[-3:]:
                output += '{0}\n\n {1}\n\n'.format(text, annotations)
            
            # Print the top and bottom 3 testing samples as well if the data has been split into subsets
            try:
                output += "Data split into {0} samples for training and {1} samples for testing.\n".format(len(self.train), len(self.validation))
                output += "Top and bottom 3 samples for testing:\n\n"

                # Output the top and bottom 3 results from the testing data
                for text, annotations in self.validation[:3] + self.validation[-3:]:
                    output += '{0}\n\n {1}\n\n'.format(text, annotations)
            except (TypeError, AttributeError) as e:
                pass
        
        elif step == 7:
            # Output after a model is successfully loaded for training
            output = "Loaded model {0}\n\n".format(self.base_model)

        elif step == 8:
            # Print loss at current epoch with training data
            output = "{0}, Losses with Training data: {1}\n".format(self.losses_train[-1][0], self.losses_train[-1][1])

        elif step == 9:
            # Print loss at current epoch with testing data
            output = "{0}, Losses with Testing data: {1}\n".format(self.losses_test[-1][0], self.losses_test[-1][1])

        elif step == 10:
            # Output after model is successfully saved to disk
            output = "\nModel successfully saved to disk at directory: {0}\n\n".format(self.path + self.model + '/')

        elif step == 11:
            # Output after evaluation metrics have been calculated
            output = "RESPONSE: {0} rows x cols\nSample Data:\n\n".format(self.response_df.shape)
            output += "{0}\n...\n{1}\n\n".format(self.response_df.head(10).to_string(), self.response_df.tail(5).to_string())

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