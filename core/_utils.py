import os
import sys
import time
import string
import numpy as np
import pandas as pd
from sklearn import preprocessing
import ServerSideExtension_pb2 as SSE

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
    This function takes in a SSE Request together with a row template and column headers as lists of strings.
    Returns a data frame for the request.
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

def scale(df, s, **kwargs):
    """
    Scale values in a Data Frame using the relevant sklearn preprocessing method
    Valid options for the scaler are: standard, minmax, maxabs, robust, quantile
    """
    
    scalers = {'standard':'StandardScaler', 'minmax':'MinMaxScaler', 'maxabs':'MaxAbsScaler',\
               'robust':'RobustScaler', 'quantile':'QuantileTransformer'}
    
    scaler = getattr(preprocessing, scalers[s])
    scaler = scaler(**kwargs)
    
    return scaler.fit_transform(df)