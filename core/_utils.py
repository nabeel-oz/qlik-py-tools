import os
import sys
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