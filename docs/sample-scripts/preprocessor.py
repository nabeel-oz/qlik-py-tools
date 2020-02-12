import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import FeatureHasher

# Custom transformer for preprocessing data based on feature definitions
class Preprocessor(TransformerMixin):
    """
    A class that preprocesses a given dataset based on feature definitions.
    This class automates One Hot Encoding, Hashing and Scaling.
    """
    
    def __init__(self, features, return_type='np', scale_hashed=True, missing="zeros", scaler="standard", **kwargs):
        """
        Initialize the Preprocessor object based on the features dataframe.
        
        The features dataframe must include these columns: Name, Variable_Type, Feature_Strategy.      
        If Feature_Strategy includes hashing, the Hash_Features column must also be included.
        The dataframe must be indexed by Name.
                
        For further information on the columns refer to the project documentation: 
        https://github.com/nabeel-qlik/qlik-py-tools
        """
        
        self.features = features
        self.return_type = return_type
        self.scale_hashed = scale_hashed
        self.missing = missing
        self.scaler = scaler
        self.kwargs = kwargs
        self.ohe = False
        self.hash = False
        self.scale = False
        self.no_prep = False
        
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
            
            # Convert Hash_Features column to integers
            self.hash_meta.loc[:,"hash_features"] = self.hash_meta.loc[:,"hash_features"].astype(np.int64)
        
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
            
            self = self.__init__(features)
        
        # Get a subset of the data that requires one hot encoding
        self.ohe_df = X[self.ohe_meta.index.tolist()]
              
        # Apply one hot encoding to relevant columns
        self.ohe_df = pd.get_dummies(self.ohe_df, columns=self.ohe_df.columns)
        
        # Keep a copy of the OHE dataframe structure so we can align the transform dataset 
        self.ohe_df_structure = pd.DataFrame().reindex_like(self.ohe_df)
        
        # Scaling needs to be fit exclusively on the training data so as not to influence the results
        if self.scale:
            # Get a subset of the data that requires scaling
            self.scale_df = X[self.scale_meta.index.tolist()]
                   
        # If hashed columns need to be scaled, these need to be considered when setting up the scaler as well
        if self.hash and self.scale_hashed:
            # Get a subset of the data that requires feature hashing
            self.hash_df = X[self.hash_meta.index.tolist()]
            hash_cols = self.hash_df.columns

            # Hash unique values for each relevant column and then join to a dataframe for hashed data
            for c in hash_cols:
                unique = self.hasher(self.hash_df, c, self.hash_meta["hash_features"].loc[c])
                self.hash_df = self.hash_df.join(unique, on=c)
                self.hash_df = self.hash_df.drop(c, axis=1)
                
            if self.scale:
                self.scale_df = self.scale_df.join(self.hash_df)
            else:
                self.scale_df = self.hash_df 
        
        if len(self.scale_df) > 0:
            # Get an instance of the sklearn scaler fit to X
            self.scaler_instance = self.get_scaler(self.scale_df, missing=self.missing, scaler=self.scaler, **self.kwargs)
        
        return self
    
    
    def transform(self, X, y=None):
        """
        Transform X with the encoding and scaling requirements set by fit().
        This function will perform One Hot Encoding, Feature Hashing and Scaling on X.
        Returns X_transform as a numpy array or a pandas dataframe based on return_type set in constructor.
        """
        
        self.X_transform = None
        
        if self.ohe:
            # Get a subset of the data that requires one hot encoding
            self.ohe_df = X[self.ohe_meta.index.tolist()]

            # Apply one hot encoding to relevant columns
            self.ohe_df = pd.get_dummies(self.ohe_df, columns=self.ohe_df.columns)

            # Align the columns with the original dataset. 
            # This is to prevent different number or order of features between training and test datasets.
            self.ohe_df = self.ohe_df.align(self.ohe_df_structure, join='right', axis=1)[0]

            # Fill missing values in the OHE dataframe, that may appear after alignment, with zeros.
            self.ohe_df = self.fillna(self.ohe_df, missing="zeros")
            
            # Add the encoded columns to the result dataset
            self.X_transform = self.ohe_df
        
        if self.hash:
            # Get a subset of the data that requires feature hashing
            self.hash_df = X[self.hash_meta.index.tolist()]
            hash_cols = self.hash_df.columns

            # Hash unique values for each relevant column and then join to a dataframe for hashed data
            for c in hash_cols:
                unique = self.hasher(self.hash_df, c, self.hash_meta["hash_features"].loc[c])
                self.hash_df = self.hash_df.join(unique, on=c)
                self.hash_df = self.hash_df.drop(c, axis=1)
        
        if self.scale:
            # Get a subset of the data that requires scaling
            self.scale_df = X[self.scale_meta.index.tolist()]

        # If scale_hashed = True join the hashed columns to the scaling dataframe
        if self.hash and self.scale_hashed:
            if self.scale:
                self.scale_df = self.scale_df.join(self.hash_df)
            else:
                self.scale_df = self.hash_df
                # If only hashed columns are being scaled, the scaler needs to be instantiated
                self.scaler_instance = self.get_scaler(self.scale_df, missing=self.missing, scaler=self.scaler, **self.kwargs)
        elif self.hash:
            # Add the hashed columns to the result dataset
            if self.X_transform is None:
                self.X_transform = self.hash_df
            else:
                self.X_transform = self.X_transform.join(self.hash_df)
        
        # Perform scaling on the relevant data
        if len(self.scale_df) > 0:
            self.scale_df = self.fillna(self.scale_df, missing=self.missing)
            self.scale_df = pd.DataFrame(self.scaler_instance.transform(self.scale_df), index=self.scale_df.index, columns=self.scale_df.columns)
            
            # Add the scaled columns to the result dataset
            if self.X_transform is None:
                self.X_transform = self.scale_df
            else:
                self.X_transform = self.X_transform.join(self.scale_df)
               
        if self.no_prep:
            # Get a subset of the data that doesn't require preprocessing
            self.no_prep_df = X[self.none_meta.index.tolist()]
        
            # Finally join the columns that do not require preprocessing to the result dataset
            if self.X_transform is None:
                self.X_transform = self.no_prep_df
            else:
                self.X_transform = self.X_transform.join(self.no_prep_df)
        
        if self.return_type == 'np':
            return self.X_transform.values
        
        return self.X_transform
    
    
    def fit_transform(self, X, y=None, features=None, retrain=False):
        """
        Apply fit() then transform()
        """
        
        if features is None:
            features = self.features
        
        return self.fit(X, y, features, retrain).transform(X, y)
    
    
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
    def get_scaler(df, missing="zeros", scaler="standard", **kwargs):
        """
        Fit a sklearn scaler on a Data Frame and return the scaler.
        Valid options for the scaler are: standard, minmax, maxabs, robust, quantile
        Missing values must be dealt with before the scaling is applied. 
        Valid options specified through the missing parameter are: zeros, mean, median, mode
        """

        scalers = {'standard':'StandardScaler', 'minmax':'MinMaxScaler', 'maxabs':'MaxAbsScaler',\
                   'robust':'RobustScaler', 'quantile':'QuantileTransformer'}

        s = getattr(preprocessing, scalers[scaler])
        s = s(**kwargs)

        df = Preprocessor.fillna(df, missing=missing)
        
        return s.fit(df)