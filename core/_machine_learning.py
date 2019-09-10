import os
import sys
import time
import copy
import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
from pathlib import Path
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Workaround for Keras issue #1406
# "Using X backend." always printed to stdout #1406 
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
sys.stderr = stderr

import _utils as utils

class PersistentModel:
    """
    A general class to manage persistent models
    """
    
    def __init__(self):
        """
        Basic contructor
        """
        
        self.name = None
        self.state = None
        self.state_timestamp = None
        self.overwrite = False
        
    def save(self, name, path, compress=3):
        """
        Save the model to disk at the specified path.
        If the model already exists and self.overwrite=False, throw an exception.
        If self.overwrite=True, replace any existing file.
        """
        
        # Create string for path and file name
        f = path + name + '.joblib'

        # Create a path for the lock file
        f_lock = f + '.lock'
                
        # Create the directory if required
        try:
            Path(path).mkdir(parents=True, exist_ok=False) 
        except FileExistsError:
            pass
        
        # If the file exists and overwriting is not allowed, raise an exception
        if Path(f).exists() and not self.overwrite:
            raise FileExistsError("The specified model name already exists: {0}.".format(name + '.joblib')\
                                  +"\nPass overwrite=True if it is ok to overwrite.")
        # Check if the file is currently locked
        elif Path(f_lock).exists():
            # Wait a few seconds and check again
            time.sleep(2)
            # If the file is still locked raise an exception
            if Path(f_lock).exists():
                raise TimeoutError("The specified model is locked. If you believe this to be wrong please delete file {0}".format(f_lock))
        else:
            # Update properties
            self.name = name
            self.state = 'saved'
            self.state_timestamp = time.time()
            
            # Create the lock file
            joblib.dump(f_lock, filename=Path(f_lock), compress=compress)

            try:
                # Store this instance to file
                joblib.dump(self, filename=Path(f), compress=compress)
            finally:
                # Delete the lock file
                Path(f_lock).unlink()
                
        return self
    
    def load(self, name, path):        
        """
        Check if the model exists at the specified path and return it to the caller.
        If the model is not found throw an exception.
        """
        
        with open(Path(path + name + '.joblib'), 'rb') as f:
            self = joblib.load(f)
        
        return self

class Preprocessor(TransformerMixin):
    """
    A class that preprocesses a given dataset based on feature definitions passed as a dataframe.
    This class automates One Hot Encoding, Hashing, Text Vectorizing and Scaling.
    """
    
    def __init__(self, features, return_type='np', scale_hashed=True, scale_vectors=True, missing="zeros", scaler="StandardScaler", logfile=None, **kwargs):
        """
        Initialize the Preprocessor object based on the features dataframe.
        
        **kwargs are keyword arguments passed to the sklearn scaler instance.

        The features dataframe must include these columns: name, variable_type, feature_strategy.      
        If Feature_Strategy includes hashing or text vectorizing, the strategy_args column must also be included.
        The dataframe must be indexed by name.
                
        For further information on the columns refer to the project documentation: 
        https://github.com/nabeel-oz/qlik-py-tools
        """
        
        self.features = features
        self.return_type = return_type
        self.scale_hashed = scale_hashed
        self.scale_vectors = scale_vectors
        self.missing = missing
        self.scaler = scaler
        self.kwargs = kwargs
        self.ohe = False
        self.hash = False
        self.cv = False
        self.tfidf = False
        self.text = False
        self.scale = False
        self.no_prep = False
        self.log = logfile
        
        # Collect features for one hot encoding
        self.ohe_meta = features.loc[features["feature_strategy"] == "one hot encoding"].copy()
        
        # Set a flag if one hot encoding will be required
        if len(self.ohe_meta) > 0:
            self.ohe = True
        
        # Collect features for hashing
        self.hash_meta = features.loc[features["feature_strategy"] == "hashing"].copy()
        
        # Set a flag if feature hashing will be required
        if len(self.hash_meta) > 0:
            self.hash = True
            
            # Convert strategy_args column to integers
            self.hash_meta.loc[:,"strategy_args"] = self.hash_meta.loc[:,"strategy_args"].astype(np.int64, errors="ignore")
        
        # Collect features for count vectorizing
        self.cv_meta = features.loc[features["feature_strategy"] == "count_vectorizing"].copy()
        
        # Set a flag if count vectorizing will be required
        if len(self.cv_meta) > 0:
            self.cv = True
            
            # Convert strategy_args column to key word arguments for the sklearn CountVectorizer class
            self.cv_meta.loc[:,"strategy_args"] = self.cv_meta.loc[:,"strategy_args"].apply(utils.get_kwargs).\
            apply(utils.get_kwargs_by_type)
        
        # Collect features for term frequency inverse document frequency (TF-IDF) vectorizing
        self.tfidf_meta = features.loc[features["feature_strategy"] == "tf_idf"].copy()
        
        # Set a flag if tfidf vectorizing will be required
        if len(self.tfidf_meta) > 0:
            self.tfidf = True
            
            # Convert strategy_args column to key word arguments for the sklearn TfidfVectorizer class
            self.tfidf_meta.loc[:,"strategy_args"] = self.tfidf_meta.loc[:,"strategy_args"].apply(utils.get_kwargs).\
            apply(utils.get_kwargs_by_type)
        
         # Collect features for text similarity one hot encoding
        self.text_meta = features.loc[features["feature_strategy"] == "text_similarity"].copy()
        
        # Set a flag if text similarity OHE will be required
        if len(self.text_meta) > 0:
            self.text = True
        
        # Collect features for scaling
        self.scale_meta = features.loc[features["feature_strategy"] == "scaling"].copy()
        
        # Set a flag if scaling will be required
        if len(self.scale_meta) > 0:
            self.scale = True
        
        # Collect other features
        self.none_meta = features.loc[features["feature_strategy"] == "none"].copy()
        
        # Set a flag if there are features that don't require preprocessing
        if len(self.none_meta) > 0:
            self.no_prep = True

        # Output information to the terminal and log file if required
        if self.log is not None:
            self._print_log(1)
    
    def fit(self, X, y=None, features=None, retrain=False):
        """
        Fit to the training dataset, storing information that will be needed for the transform dataset.
        Return the Preprocessor object.
        Optionally re-initizialise the object by passing retrain=True, and resending the features dataframe
        """

        # Reinitialize this Preprocessor instance if required
        if retrain:
            if features is None:
                features = self.features
            
            self.__init__(features)
        
        # Set up an empty data frame for data to be scaled
        scale_df = pd.DataFrame()

        ohe_df = None
        hash_df = None
        cv_df = None
        tfidf_df = None
        text_df = None
        
        if self.ohe:
            # Get a subset of the data that requires one hot encoding
            ohe_df = X[self.ohe_meta.index.tolist()]
                
            # Apply one hot encoding to relevant columns
            ohe_df = pd.get_dummies(ohe_df, columns=ohe_df.columns)
            
            # Keep a copy of the OHE dataframe structure so we can align the transform dataset 
            self.ohe_df_structure = pd.DataFrame().reindex_like(ohe_df)
        
        # Scaling needs to be fit exclusively on the training data so as not to influence the results
        if self.scale:
            # Get a subset of the data that requires scaling
            scale_df = X[self.scale_meta.index.tolist()]
                   
        if self.hash:
            # Get a subset of the data that requires feature hashing
            hash_df = X[self.hash_meta.index.tolist()]
            hash_cols = hash_df.columns

            # Hash unique values for each relevant column and then join to a dataframe for hashed data
            for c in hash_cols:
                unique = self.hasher(hash_df, c, self.hash_meta["strategy_args"].loc[c])
                hash_df = hash_df.join(unique, on=c)
                hash_df = hash_df.drop(c, axis=1)

            # If hashed columns need to be scaled, these need to be considered when setting up the scaler as well    
            if self.scale_hashed:
                if self.scale:
                    scale_df = scale_df.join(hash_df)
                else:
                    scale_df = hash_df 
        
        if self.cv:
            # Get a subset of the data that requires count vectorizing
            cv_df = X[self.cv_meta.index.tolist()]
            cv_cols = cv_df.columns

            # Get count vectors for each relevant column and then join to a dataframe for count vectorized data
            for c in cv_cols:
                unique = self.text_vectorizer(cv_df, c, type="count", **self.cv_meta["strategy_args"].loc[c])
                cv_df = cv_df.join(unique, on=c)
                cv_df = cv_df.drop(c, axis=1)

            # Keep a copy of the count vectorized dataframe structure so we can align the transform dataset 
            self.cv_df_structure = pd.DataFrame().reindex_like(cv_df)

            # If text vector columns need to be scaled, these need to be considered when setting up the scaler as well    
            if self.scale_vectors:
                if self.scale or (self.scale_hashed and self.hash):
                    scale_df = scale_df.join(cv_df)
                else:
                    scale_df = cv_df 

        if self.tfidf:
            # Get a subset of the data that requires tfidf vectorizing
            tfidf_df = X[self.tfidf_meta.index.tolist()]
            tfidf_cols = tfidf_df.columns

            # Get tfidf vectors for each relevant column and then join to a dataframe for tfidf vectorized data
            for c in tfidf_cols:
                unique = self.text_vectorizer(tfidf_df, c, type="tfidf", **self.tfidf_meta["strategy_args"].loc[c])
                tfidf_df = tfidf_df.join(unique, on=c)
                tfidf_df = tfidf_df.drop(c, axis=1)

            # Keep a copy of the tfidf vectorized dataframe structure so we can align the transform dataset 
            self.tfidf_df_structure = pd.DataFrame().reindex_like(tfidf_df)
            
            # If text vector columns need to be scaled, these need to be considered when setting up the scaler as well    
            if self.scale_vectors:
                if self.scale or (self.scale_hashed and self.hash) or self.cv:
                    scale_df = scale_df.join(tfidf_df)
                else:
                    scale_df = tfidf_df 
        
        if self.text:
            # Get a subset of the data that requires text similarity OHE
            text_df = X[self.text_meta.index.tolist()]
            text_cols = text_df.columns

            # Get text similarity OHE for each relevant column and then join to a dataframe for text similarity OHE data
            for c in text_cols:
                unique = self.text_similarity(text_df, c)
                text_df = text_df.join(unique, on=c)
                text_df = text_df.drop(c, axis=1)

            # Keep a copy of the text similarity OHE dataframe structure so we can align the transform dataset 
            self.text_df_structure = pd.DataFrame().reindex_like(text_df)

        try:
            if len(scale_df) > 0:
                # Get an instance of the sklearn scaler fit to X
                self.scaler_instance = self.get_scaler(scale_df, missing=self.missing, scaler=self.scaler, **self.kwargs)

                # Keep a copy of the scaling dataframe structure so we can align the transform dataset 
                self.scale_df_structure = pd.DataFrame().reindex_like(scale_df)
        except AttributeError:
            pass

        # Output information to the terminal and log file if required
        if self.log is not None:
            self._print_log(2, ohe_df=ohe_df, scale_df=scale_df, hash_df=hash_df, cv_df=cv_df, tfidf_df=tfidf_df, text_df=text_df)

        return self
      
    def transform(self, X, y=None):
        """
        Transform X with the encoding and scaling requirements set by fit().
        This function will perform One Hot Encoding, Feature Hashing and Scaling on X.
        Returns X_transform as a numpy array or a pandas dataframe based on return_type set in constructor.
        """        
        
        X_transform = None
        scale_df = pd.DataFrame() # Initialize as empty Data Frame for convenience of concat operations below
        ohe_df = None
        hash_df = None
        cv_df = None
        tfidf_df = None
        text_df = None
        
        if self.ohe:
            # Get a subset of the data that requires one hot encoding
            ohe_df = X[self.ohe_meta.index.tolist()]

            # Apply one hot encoding to relevant columns
            ohe_df = pd.get_dummies(ohe_df, columns=ohe_df.columns)

            # Align the columns with the original dataset. 
            # This is to prevent different number or order of features between training and test datasets.
            ohe_df = ohe_df.align(self.ohe_df_structure, join='right', axis=1)[0]

            # Fill missing values in the OHE dataframe, that may appear after alignment, with zeros.
            ohe_df = self.fillna(ohe_df, missing="zeros")
            
            # Add the encoded columns to the result dataset
            X_transform = ohe_df

        if self.hash:
            # Get a subset of the data that requires feature hashing
            hash_df = X[self.hash_meta.index.tolist()]
            hash_cols = hash_df.columns

            # Hash unique values for each relevant column and then join to a dataframe for hashed data
            for c in hash_cols:
                unique = self.hasher(hash_df, c, self.hash_meta["strategy_args"].loc[c])
                hash_df = hash_df.join(unique, on=c)
                hash_df = hash_df.drop(c, axis=1)
                # Fill any missing values in the hash dataframe
                hash_df = self.fillna(hash_df, missing="zeros")
        
        if self.cv:
            # Get a subset of the data that requires count vectorizing
            cv_df = X[self.cv_meta.index.tolist()]
            cv_cols = cv_df.columns

            # Get count vectors for each relevant column and then join to a dataframe for count vectorized data
            for c in cv_cols:
                unique = self.text_vectorizer(cv_df, c, type="count", **self.cv_meta["strategy_args"].loc[c])
                cv_df = cv_df.join(unique, on=c)
                cv_df = cv_df.drop(c, axis=1)

            # Align the columns with the original dataset. 
            # This is to prevent different number or order of features between training and test datasets.
            cv_df = cv_df.align(self.cv_df_structure, join='right', axis=1)[0]

            # Fill missing values in the dataframe that may appear after alignment with zeros.
            cv_df = self.fillna(cv_df, missing="zeros")

        if self.tfidf:
            # Get a subset of the data that requires tfidf vectorizing
            tfidf_df = X[self.tfidf_meta.index.tolist()]
            tfidf_cols = tfidf_df.columns

            # Get tfidf vectors for each relevant column and then join to a dataframe for tfidf vectorized data
            for c in tfidf_cols:
                unique = self.text_vectorizer(tfidf_df, c, type="tfidf", **self.tfidf_meta["strategy_args"].loc[c])
                tfidf_df = tfidf_df.join(unique, on=c)
                tfidf_df = tfidf_df.drop(c, axis=1)

            # Align the columns with the original dataset. 
            # This is to prevent different number or order of features between training and test datasets.
            tfidf_df = tfidf_df.align(self.tfidf_df_structure, join='right', axis=1)[0]

            # Fill missing values in the dataframe that may appear after alignment with zeros.
            tfidf_df = self.fillna(tfidf_df, missing="zeros")
        
        if self.text:
            # Get a subset of the data that requires text similarity OHE
            text_df = X[self.text_meta.index.tolist()]
            text_cols = text_df.columns

            # Get text similarity OHE for each relevant column and then join to a dataframe for text similarity OHE data
            for c in text_cols:
                unique = self.text_similarity(text_df, c)
                text_df = text_df.join(unique, on=c)
                text_df = text_df.drop(c, axis=1)

            # Align the columns with the original dataset. 
            # This is to prevent different number or order of features between training and test datasets.
            text_df = text_df.align(self.text_df_structure, join='right', axis=1)[0]

            # Fill missing values in the dataframe that may appear after alignment with zeros.
            text_df = self.fillna(text_df, missing="zeros")

            # Add the text similary OHE data to the result dataset
            if X_transform is None:
                X_transform = text_df
            else:
                X_transform = pd.concat([X_transform, text_df], join='outer', axis=1, sort=False)

        if self.scale:
            # Get a subset of the data that requires scaling
            scale_df = X[self.scale_meta.index.tolist()]

        # If scale_hashed = True join the hashed columns to the scaling dataframe
        if self.hash and self.scale_hashed:
            if self.scale:
                scale_df = pd.concat([scale_df, hash_df], join='outer', axis=1, sort=False)
            else:
                scale_df = hash_df
                # If only hashed columns are being scaled, the scaler needs to be instantiated
                self.scaler_instance = self.get_scaler(scale_df, missing=self.missing, scaler=self.scaler, **self.kwargs)
        elif self.hash:
            # Add the hashed columns to the result dataset
            if X_transform is None:
                X_transform = hash_df
            else:
                X_transform = pd.concat([X_transform, hash_df], join='outer', axis=1, sort=False)

        # If scale_vectors = True join the count vectorized columns to the scaling dataframe
        if self.cv and self.scale_vectors:
            if self.scale or (self.hash and self.scale_hashed):
                scale_df = pd.concat([scale_df, cv_df], join='outer', axis=1, sort=False)
            else:
                scale_df = cv_df
                # If only count vectorized columns are being scaled, the scaler needs to be instantiated
                self.scaler_instance = self.get_scaler(scale_df, missing=self.missing, scaler=self.scaler, **self.kwargs)
        elif self.cv:
            # Add the count vectorized columns to the result dataset
            if X_transform is None:
                X_transform = cv_df
            else:
                X_transform = pd.concat([X_transform, cv_df], join='outer', axis=1, sort=False)

        # If scale_vectors = True join the tfidf vectorized columns to the scaling dataframe
        if self.tfidf and self.scale_vectors:
            if self.scale or (self.hash and self.scale_hashed) or self.cv:
                scale_df = pd.concat([scale_df, tfidf_df], join='outer', axis=1, sort=False)
            else:
                scale_df = tfidf_df
                # If only tfidf vectorized columns are being scaled, the scaler needs to be instantiated
                self.scaler_instance = self.get_scaler(scale_df, missing=self.missing, scaler=self.scaler, **self.kwargs)
        elif self.tfidf:
            # Add the count vectorized columns to the result dataset
            if X_transform is None:
                X_transform = tfidf_df
            else:
                X_transform = pd.concat([X_transform, tfidf_df], join='outer', axis=1, sort=False)

        try:
            # Perform scaling on the relevant data
            if len(scale_df) > 0:
                # Align the columns with the original dataset. 
                # This is to prevent different number or order of features between training and test datasets.
                scale_df = scale_df.align(self.scale_df_structure, join='right', axis=1)[0]
                
                scale_df = self.fillna(scale_df, missing=self.missing)

                scale_df = pd.DataFrame(self.scaler_instance.transform(scale_df), index=scale_df.index, columns=scale_df.columns)
                
                # Add the scaled columns to the result dataset
                if X_transform is None:
                    X_transform = scale_df
                else:
                    X_transform = pd.concat([X_transform, scale_df], join='outer', axis=1, sort=False)
        except AttributeError:
            pass

        if self.no_prep:
            # Get a subset of the data that doesn't require preprocessing
            no_prep_df = X[self.none_meta.index.tolist()]
            # Fill any missing values in the no prep dataframe
            no_prep_df = self.fillna(no_prep_df, missing="zeros")
        
            # Finally join the columns that do not require preprocessing to the result dataset
            if X_transform is None:
                X_transform = no_prep_df
            else:
                X_transform = pd.concat([X_transform, no_prep_df], join='outer', axis=1, sort=False)
        
        # Output information to the terminal and log file if required
        if self.log is not None:
            self._print_log(3, ohe_df=ohe_df, scale_df=scale_df, hash_df=hash_df, cv_df=cv_df, tfidf_df=tfidf_df, text_df=text_df, X_transform=X_transform)

        if self.return_type == 'np':
            return X_transform.values
        
        return X_transform
    
    def fit_transform(self, X, y=None, features=None, retrain=False):
        """
        Apply fit() then transform()
        """
        
        if features is None:
            features = self.features
        
        return self.fit(X, y, features, retrain).transform(X, y)
    
    def _print_log(self, step, **kwargs):
        """
        Output useful information to stdout and the log file if debugging is required.
        step: Print the corresponding step in the log
        kwargs: dictionary of dataframes to be used in the log
        """
        
        if step == 1:
            if self.ohe:
                sys.stdout.write("Features for one hot encoding: \n{0}\n\n".format(self.ohe_meta))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Features for one hot encoding: \n{0}\n\n".format(self.ohe_meta))
            
            if self.hash:
                sys.stdout.write("Features for hashing: \n{0}\n\n".format(self.hash_meta))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Features for hashing: \n{0}\n\n".format(self.hash_meta))
            
            if self.cv:
                sys.stdout.write("Features for count vectorization: \n{0}\n\n".format(self.cv_meta))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Features for count vectorization: \n{0}\n\n".format(self.cv_meta))
            
            if self.tfidf:
                sys.stdout.write("Features for tfidf vectorization: \n{0}\n\n".format(self.tfidf_meta))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Features for tfidf vectorization: \n{0}\n\n".format(self.tfidf_meta))
            
            if self.scale:
                sys.stdout.write("Features for scaling: \n{0}\n\n".format(self.scale_meta))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Features for scaling: \n{0}\n\n".format(self.scale_meta))

        elif step == 2:
            if self.ohe:
                sys.stdout.write("Fit ohe_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['ohe_df'].shape, kwargs['ohe_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Fit ohe_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['ohe_df'].shape, kwargs['ohe_df'].head()))
            
            if self.hash:
                sys.stdout.write("Fit hash_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['hash_df'].shape, kwargs['hash_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Fit hash_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['hash_df'].shape, kwargs['hash_df'].head()))
            
            if self.cv:
                sys.stdout.write("Fit cv_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['cv_df'].shape, kwargs['cv_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Fit cv_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['cv_df'].shape, kwargs['cv_df'].head()))
            
            if self.tfidf:
                sys.stdout.write("Fit tfidf_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['tfidf_df'].shape, kwargs['tfidf_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Fit tfidf_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['tfidf_df'].shape, kwargs['tfidf_df'].head()))
            
            try:
                if len(kwargs['scale_df']) > 0:
                    sys.stdout.write("Fit scale_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['scale_df'].shape, kwargs['scale_df'].head()))
                    
                    with open(self.log,'a', encoding='utf-8') as f:
                        f.write("Fit scale_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['scale_df'].shape, kwargs['scale_df'].head()))
            except AttributeError:
                pass
        
        elif step == 3:
            if self.ohe:
                sys.stdout.write("Transform ohe_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['ohe_df'].shape, kwargs['ohe_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Transform ohe_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['ohe_df'].shape, kwargs['ohe_df'].head()))
            
            if self.hash:
                sys.stdout.write("Transform hash_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['hash_df'].shape, kwargs['hash_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Transform hash_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['hash_df'].shape, kwargs['hash_df'].head()))
            
            if self.cv:
                sys.stdout.write("Transform cv_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['cv_df'].shape, kwargs['cv_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Transform cv_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['cv_df'].shape, kwargs['cv_df'].head()))
            
            if self.tfidf:
                sys.stdout.write("Transform tfidf_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['tfidf_df'].shape, kwargs['tfidf_df'].head()))
                
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("Transform tfidf_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['tfidf_df'].shape, kwargs['tfidf_df'].head()))
            
            try:
                if len(kwargs['scale_df']) > 0:
                    sys.stdout.write("Transform scale_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['scale_df'].shape, kwargs['scale_df'].head()))
                    
                    with open(self.log,'a', encoding='utf-8') as f:
                        f.write("Transform scale_df shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['scale_df'].shape, kwargs['scale_df'].head()))
            except AttributeError:
                pass

            try:
                sys.stdout.write("X_transform shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['X_transform'].shape, kwargs['X_transform'].head()))
                    
                with open(self.log,'a', encoding='utf-8') as f:
                    f.write("X_transform shape:{0}\nSample Data:\n{1}\n\n".format(kwargs['X_transform'].shape, kwargs['X_transform'].head()))
            except AttributeError:
                pass

    @staticmethod
    def hasher(df, col, n_features):
        """
        Hash the unique values in the specified column in the given dataframe, creating n_features
        """
        
        unique = pd.DataFrame(df[col].unique(), columns=[col])
        fh = FeatureHasher(n_features=n_features, input_type="string")
        hashed = fh.fit_transform(unique.loc[:, col])
        unique = unique.join(pd.DataFrame(hashed.toarray()).add_prefix(col))
        return unique.set_index(col)
    
    @staticmethod
    def text_vectorizer(df, col, type="count", **kwargs):
        """
        Create count vectors using the sklearn TfidfVectorizer or CountVectorizer for the specified column in the given dataframe.
        The type argument can be "tfidf" referring to TfidfVectorizer, anything else defaults to CountVectorizer.
        """
        
        unique = pd.DataFrame(df[col].unique(), columns=[col])
        
        if type == "tfidf":
            v = TfidfVectorizer(**kwargs)
        else:
            v = CountVectorizer(**kwargs)
        
        vectorized = v.fit_transform(unique.loc[:, col])

        feature_names = v.get_feature_names()
        col_names = []

        for i,j in enumerate(feature_names):
            col_names.append("{}_{}".format(i,j))

        unique = unique.join(pd.DataFrame(vectorized.toarray(), columns=col_names).add_prefix(col+"_"))
        return unique.set_index(col)
    
    @staticmethod
    def text_similarity(df, col):
        """
        Convert strings to their unicode representation and then apply one hot encoding, creating one feature for each unique character in the column. 
        This can be useful when similarity between strings is significant.
        """
        
        unique = pd.DataFrame(df[col].unique(), columns=[col])
        
        encoded = pd.DataFrame(unique.loc[:,col].apply(lambda s: [ord(a) for a in s]), index=unique.index)
        
        mlb = preprocessing.MultiLabelBinarizer()
        encoded = pd.DataFrame(mlb.fit_transform(encoded[col]),columns=mlb.classes_, index=encoded.index).add_prefix(col+"_")
        
        unique = unique.join(encoded)
        
        return unique.set_index(col)

    @staticmethod
    def fillna(df, missing="zeros"):
        """
        Fill empty values in a Data Frame with the chosen method.
        Valid options for missing are: zeros, mean, median, mode
        """

        if missing == "mean":
            return df.fillna(df.mean())
        elif missing == "median":
            return df.fillna(df.median())
        elif missing == "mode":
            return df.fillna(df.mode().iloc[0])
        elif missing == "none":
            return df
        else:
            return df.fillna(0)
    
    @staticmethod
    def get_scaler(df, missing="zeros", scaler="StandardScaler", **kwargs):
        """
        Fit a sklearn scaler on a Data Frame and return the scaler.
        Valid options for the scaler are: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
        Missing values must be dealt with before the scaling is applied. 
        Valid options specified through the missing parameter are: zeros, mean, median, mode
        """

        s = getattr(preprocessing, scaler)
        s = s(**kwargs)

        df = Preprocessor.fillna(df, missing=missing)
        
        return s.fit(df)       

class Reshaper(TransformerMixin):
    """
    A class that reshapes the feature matrix based on the input_shape.
    This class is built for Keras estimators where recurrent and convolutional layers can required 3D or 4D inputs.
    It is meant to be used after preprocessing and before fitting the estimator.
    """

    def __init__(self, first_layer_kwargs=None, lags=None, logfile=None, **kwargs):
        """
        Initialize the Reshaper with the Keras model first layer's kwargs.
        Additionally take in the number of lag observations to be used in reshaping the data.
        first_layer_kwargs is updated during fit and transform, so it must be a reference to the kwargs used to build the Keras model.
        Optional arguments are a logfile to output debug info.
        """

        self.first_layer_kwargs = first_layer_kwargs
        self.lags = lags
        self.logfile = logfile
    
    def fit(self, X, y=None):
        """
        Update the input shape based on the number of features in X.
        Return this Reshaper object.
        """

        # If it has not been specified, it is simply the number of features in X
        if 'input_shape' not in self.first_layer_kwargs:
            self.first_layer_kwargs['input_shape'] = tuple([X.shape[1]])
        # Else update the input shape based on the number of features after preprocessing
        else:
            # Transform to a list to make the input_shape mutable
            self.first_layer_kwargs['input_shape'] = list(self.first_layer_kwargs['input_shape'])
            # Update the number of features based on X
            self.first_layer_kwargs['input_shape'][-1] = X.shape[1]
            # Transform back to a tuple as required by Keras
            self.first_layer_kwargs['input_shape'] = tuple(self.first_layer_kwargs['input_shape'])
        
        self.input_shape = self.first_layer_kwargs['input_shape']

        # Debug information is printed to the terminal and logs if required
        if self.logfile:
            self._print_log(1)

        return self
    
    def transform(self, X, y=None):
        """
        Apply the new shape to the data provided in X.
        X is expected to be a 2D DataFrame of samples and features.
        If self.lags is an integer, previous samples will be used as lag observations and added as input for each sample.
        The lags parameter is expected when using a 3D or 4D input shape.
        """
        
        # If X is n_samples by n_features and no lags need to be added, we have nothing to do here
        if (len(self.input_shape) == 1) and not self.lags:
            return X
        
        # Get the lag observations and add to X
        if self.lags:
            # Add the lag observations
            X_transform = utils.add_lags(X, lag=self.lags, extrapolate=1, dropna=True)

            # Debug information is printed to the terminal and logs if required
            if self.logfile:
                self._print_log(2, data=X_transform)

        # 2D, 3D and 4D data is valid. 
        # As per the Keras convention, samples should not be specified in the input shape.
        # e.g. The input_shape can be a tuple of (subsequences, timesteps, features), with subsequences and timesteps as optional.
        # A 5D shape may be valid for e.g. a ConvLSTM with (timesteps, rows, columns, features) 
        if len(self.input_shape) > 4:
                err = "Unsupported input_shape: {}".format(self.input_shape)
                raise Exception(err)
       
        # Reshape the data
        elif len(self.input_shape) > 1:
            # Reshape input data using numpy
            X_transform = X_transform.values.reshape(self.input_shape)

            # Debug information is printed to the terminal and logs if required
            if self.logfile:
                self._print_log(3, data=X_transform)

        # Update the original input_shape with the final number of features if necessary
        # This is expected if no lag observations were added during transform, yet the data was reshaped
        if self.input_shape[-1] != X_transform.shape[-1]:
            self.fit(X_transform, y)

        return X_transform
    
    def _print_log(self, step, data=None):
        """
        Print debug info to the log
        """
        
        # Set mode to append to log file
        mode = 'a'

        if step == 1:
            # Output the updated input shape
            output = "Input shape of the data: {0}\n\n".format(self.first_layer_kwargs['input_shape'])
        elif step == 2:
            # Output sample data after adding lag observations
            output = "Lag observations added ({0} per sample).\nSample Data:\n{1}\n\n".format(self.lags, data.head())
        elif step == 3:
            # Output sample data after reshaping
            output = "Input data reshaped to {0}.\nSample Data:\n{1}\n\n".format(data.shape, data[:5])

        sys.stdout.write(output)
        with open(self.logfile, mode, encoding='utf-8') as f:
            f.write(output)


class KerasClassifierForQlik(KerasClassifier):
    """
    A subclass of the KerasClassifier Scikit-Learn wrapper.
    This class takes in a compiled Keras model as part of sk_params and uses the __call__ method as the default build_fn.
    It also stores a histories dataframe to provide metrics for each time the model is fit.
    """
      
    def __init__(self, **sk_params):
        """
        Initialize the KerasClassifierForQlik.
        The compiled Keras model should be included in sk_params under the 'neural_net' keyword argument.
        """
        
        # Deep copy sk_params so that popping build_fn does not affect subsequent instances of the estimator
        self.sk_params = sk_params
        
        # Set build_fn to the function supplied in sk_params
        self.build_fn = self.sk_params.pop('build_fn')

        # DataFrame to contain history of every training cycle
        # This DataFrame will provide metrics such as loss for each run of the fit method
        # Columns will be ['iteration', 'epoch', 'loss'] and any other metrics being calculated during training
        self.histories = pd.DataFrame()
        
        # Check the parameters using the super class method
        self.check_params(self.sk_params)  

    def get_params(self, **params):
        """Gets parameters for this estimator.
        # Arguments
            **params: ignored (exists for API compatibility).
        # Returns
            Dictionary of parameter names mapped to their values.
        """
        res = self.sk_params
        res.update({'build_fn': self.build_fn})
        return res

    def fit(self, x, y, sample_weight=None, **kwargs):
        """
        Call the super class' fit method and store metrics from the history.
        """
        
        # Fit the model to the data and store information on the training
        history = super().fit(x, y, sample_weight, **kwargs)

        # Set up a data frame with the epochs and a counter to track multiple histories
        history_df = pd.DataFrame({'iteration': len(self.histories), 'epoch': history.epoch})
        
        # Add a column per metric for each epoch e.g. loss, acc
        for key in history.history:
            history_df[key] = pd.Series(history.history[key])

        # Concatenate results from the training to the history data frame
        self.histories = pd.concat([self.histories, history_df], sort=True).sort_values(by=['iteration', 'epoch']).reset_index(drop=True)

        return history

class KerasRegressorForQlik(KerasRegressor):
    """
    A subclass of the KerasRegressor Scikit-Learn wrapper.
    This class takes in a compiled Keras model as part of sk_params and uses the __call__ method as the default build_fn.
    It also stores a histories dataframe to provide metrics for each time the model is fit.
    """
      
    def __init__(self, **sk_params):
        """
        Initialize the KerasRegressorForQlik.
        The compiled Keras model should be included in sk_params under the 'neural_net' keyword argument.
        """

        # Deep copy sk_params so that popping build_fn does not affect subsequent instances of the estimator
        self.sk_params = sk_params
        
        # Set build_fn to the function supplied in sk_params
        self.build_fn = self.sk_params.pop('build_fn')
        
        # DataFrame to contain history of every training cycle
        # This DataFrame will provide metrics such as loss for each run of the fit method
        # Columns will be ['iteration', 'epoch', 'loss'] and any other metrics being calculated during training
        self.histories = pd.DataFrame()

        # Check the parameters using the super class method
        self.check_params(self.sk_params)

    def get_params(self, **params):
        """
        Gets parameters for this estimator.
        Overrides super class method for compatibility with sklearn cross_validate.
        """

        res = self.sk_params
        res.update({'build_fn': self.build_fn})
        return res

    def fit(self, x, y, **kwargs):
        """
        Call the super class' fit method and store metrics from the history.
        """

        # Fit the model to the data and store information on the training
        history = super().fit(x, y, **kwargs)

        # Set up a data frame with the epochs and a counter to track multiple histories
        history_df = pd.DataFrame({'iteration': len(self.histories), 'epoch': history.epoch})
        
        # Add a column per metric for each epoch e.g. loss
        for key in history.history:
            history_df[key] = pd.Series(history.history[key])

        # Concatenate results from the training to the history data frame
        self.histories = pd.concat([self.histories, history_df], sort=True).sort_values(by=['iteration', 'epoch']).reset_index(drop=True)

        return history