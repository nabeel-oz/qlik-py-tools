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

def request_df(request, row_template, col_headers):
    """
    This function takes in a SSE request as a list together with a row template and column headers as lists of strings.
    Returns a Data Frame for the request.
    e.g. request_df(request, ['strData', 'numData', 'strData'], ['dim1', 'measure', 'kwargs'])
    """
    
    rows = [row for request_rows in request for row in request_rows.rows]
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
    
    strategy = method.lower()
    
    if strategy == "mean":
        return df.fillna(df.mean())
    elif strategy == "median":
        return df.fillna(df.median())
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    else
        return df.fillna(0)
    
def scale(df, s="robust", **kwargs):
    """
    Scale values in a Data Frame using the relevant sklearn preprocessing method.
    Valid options for the scaler are: standard, minmax, maxabs, robust, quantile
    """
    
    scalers = {'standard':'StandardScaler', 'minmax':'MinMaxScaler', 'maxabs':'MaxAbsScaler',\
               'robust':'RobustScaler', 'quantile':'QuantileTransformer'}
    
    scaler = getattr(preprocessing, scalers[s])
    scaler = scaler(**kwargs)
    
    return scaler.fit_transform(df)
