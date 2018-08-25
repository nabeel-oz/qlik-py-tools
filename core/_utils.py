import os
import sys
import ast
import string
import numpy as np
import pandas as pd
from sklearn import preprocessing
import ServerSideExtension_pb2 as SSE

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

def _string_to_float(s):
    """
    This function returns a float for the string parameter s, or None in the case of a ValueError exception
    """
    try:
        f = float(s)
        return f
    except ValueError:
        return None

def request_df(request_list, row_template, col_headers):
    """
    This function takes in a SSE request as a list together with a row template and column headers as lists of strings.
    Returns a Data Frame for the request.
    e.g. request_df(request_list, ['strData', 'numData', 'strData'], ['dim1', 'measure', 'kwargs'])
    """
    
    rows = [row for request_rows in request_list for row in request_rows.rows]
    outer = []
    
    for i in range(len(rows)):
        inner = []
        
        for j in range(len(row_template)):
            
            inner.append(getattr(rows[i].duals[j], row_template[j]))
        
        outer.append(inner)
    
    return pd.DataFrame(outer, columns=col_headers)

def fillna(df, method="zeros"):
    """
    Fill empty values in a Data Frame with the chosen method.
    Valid options for method are: zeros, mean, median, mode
    """

    if method == "mean":
        return df.fillna(df.mean())
    elif method == "median":
        return df.fillna(df.median())
    elif method == "mode":
        return df.fillna(df.mode().iloc[0])
    elif method == "none":
        return df
    else:
        return df.fillna(0)
    
def scale(df, missing="zeros", scaler="robust", **kwargs):
    """
    Scale values in a Data Frame using the relevant sklearn preprocessing method.
    Valid options for the scaler are: standard, minmax, maxabs, robust, quantile
    Missing values must be dealt with before the scaling is applied. 
    Valid options specified through the missing parameter are: zeros, mean, median, mode
    """
    
    scalers = {'standard':'StandardScaler', 'minmax':'MinMaxScaler', 'maxabs':'MaxAbsScaler',\
               'robust':'RobustScaler', 'quantile':'QuantileTransformer'}
    
    s = getattr(preprocessing, scalers[scaler])
    s = s(**kwargs)
    
    df = fillna(df, method=missing)
    df = pd.DataFrame(s.fit_transform(df), index=df.index, columns=df.columns)
    
    return df

def count_placeholders(series):
    """
    Count the number of null or zero values at the bottom of a series.
    """
    count = 0

    for i in range(series.size-1, -1, -1):
        if pd.isnull(series[i]) or series[i] == 0:
            count += 1
        else:
            break

    return count

def get_kwargs(str_kwargs):
    """
    Take in a string of key word arguments and return as a dictionary of key, value pairs
    The string should be in the form: 'arg1=value1,arg2=value2'
    """
    
    # Remove any extra spaces and trailing commas
    args = str_kwargs.strip()
    if args[-1] == ',':
        args = args[:-1]
    
    # The parameter and values are transformed into key value pairs
    args = args.translate(str.maketrans('', '', string.whitespace)).split(",")
    kwargs = dict([arg.split("=") for arg in args])

    # Make sure the key words are in lower case
    kwargs = {k.lower(): v for k, v in kwargs.items()}
    
    return kwargs

def get_kwargs_by_type(dict_kwargs):
    """
    Take in a dictionary of keyword arguments where values are converted to the specified data type.
    The values in the dictionary should be a string of the form: "value|type" 
    e.g. {"arg1": "2|int", "arg2": "2.0|float", "arg3": "True|bool", "arg4": "string|str"}
    """
    
    # Dictionary used to convert argument values to the correct type
    types = {"boolean":ast.literal_eval, "bool":ast.literal_eval, "integer":int, "int":int,\
             "float":float, "float":float, "string":str, "str":str}
    
    result_dict = {}
    
    # Fill up the dictionary with the keyword arguments
    for k, v in dict_kwargs.items():
        # Split the value and type
        v, t = v.split("|")
        # Convert the value based on the correct type
        result_dict[k] = types[t](v)
    
    return result_dict

def convert_types(n_samples, features_df):
    """
    Convert data in n_samples to the correct data type based on the feature definitions.
    Both parameters must be supplied as dataframes. The columns in n_samples must be equal to rows in features_df.
    The features_df dataframe must have a "name" and a "data_type" column.
    Accepted data_types are int, float, str, bool.
    """
    
    # Transpose the features dataframe and keep the data_types for each feature
    features_df_t = features_df.T
    features_df_t.columns = features_df_t.loc["name",:].tolist()
    dtypes = features_df_t.loc["data_type",:]
    
    # Dictionary used to convert argument values to the correct type
    types = {"boolean":ast.literal_eval, "bool":ast.literal_eval, "integer":int, "int":int,\
             "float":float, "float":float, "string":str, "str":str}
    
    # Convert columns by the corresponding data type
    for col in n_samples.columns:
        # Handle conversion from string to boolean
        if dtypes[col] == "boolean" or dtypes[col] == "bool":
            n_samples.loc[:, col] = n_samples.loc[:, col].astype("str").apply(str.capitalize)
        
        # Convert this column to the correct type
        n_samples.loc[:, col] = n_samples.loc[:, col].apply(types[dtypes[col]])     
        
    return n_samples