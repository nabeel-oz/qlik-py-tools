import os
import sys
import ast
import string
import locale
import warnings
import numpy as np
import pandas as pd
import ServerSideExtension_pb2 as SSE

# Suppress warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn import preprocessing

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'generated'))

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

def get_response_rows(response, template):
    """
    Take in a list of responses and covert them to SSE.Rows based on the column type specified in template
    The template should be a list of the form: ["str", "num", "dual", ...]
    For string values use: "str"
    For numeric values use: "num"
    For dual values: "dual"
    """

    response_rows = []
    
    # For each row in the response list
    for row in response:
        i = 0
        this_row = []
        
        if len(template) > 1:        
            # For each column in the row
            for col in row:
                # Convert values to type SSE.Dual according to the template list
                if template[i] == "str":
                    if col is None:
                        col = "\x00"
                    elif type(col) is not str:
                        col = "{0:.5f}".format(col)
                    this_row.append(SSE.Dual(strData=col))
                elif template[i] == "num":
                    this_row.append(SSE.Dual(numData=col))
                elif template[i] == "dual":
                    this_row.append(SSE.Dual(strData=col, numData=col))
                i = i + 1
        else:
            # Convert values to type SSE.Dual according to the template list
            if template[0] == "str":
                if row is None:
                    row = "\x00"
                elif type(row) is not str:
                    row = "{0:.5f}".format(row)
                this_row.append(SSE.Dual(strData=row))
            elif template[0] == "num":
                this_row.append(SSE.Dual(numData=row))
            elif template[0] == "dual":
                this_row.append(SSE.Dual(strData=row, numData=row))
        
        # Group columns into a iterable and add to the the response_rows
        response_rows.append(iter(this_row))

    # Values are then structured as SSE.Rows
    response_rows = [SSE.Row(duals=duals) for duals in response_rows]

    return response_rows

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

def get_scaler(df, missing="zeros", scaler="StandardScaler", **kwargs):
    """
    Fit a sklearn scaler on a Data Frame and return the scaler.
    Valid options for the scaler are: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
    Missing values must be dealt with before the scaling is applied. 
    Valid options specified through the missing parameter are: zeros, mean, median, mode
    """

    s = getattr(preprocessing, scaler)
    s = s(**kwargs)

    df = fillna(df, method=missing)
    
    return s.fit(df)  
    
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

    # If the argument string is empty return an empty dict
    if len(str_kwargs) == 0:
        return dict()
    
    # Remove any extra spaces and trailing commas
    args = str_kwargs.strip()
    if args[-1] == ',':
        args = args[:-1]
    
    # The parameter and values are transformed into key value pairs
    args = args.translate(str.maketrans('', '', string.whitespace)).split(",")
    kwargs = dict([arg.split("=") for arg in args])
    
    return kwargs

def get_kwargs_by_type(dict_kwargs):
    """
    Take in a dictionary of keyword arguments where values are converted to the specified data type.
    The values in the dictionary should be a string of the form: "value|type"
    e.g. {"arg1": "2|int", "arg2": "2.0|float", "arg3": "True|bool", "arg4": "string|str"}
    Dictionaries, lists and arrays are allowed with the following format:
    "x:1;y:2|dict|str|int" where str is the type for keys and int is the type for values
    "x;y;z|array|str" where str is the type of values in the array
    "1;2;3|list|int" where int is the type of the values in the list
    "0;1|tuple|int" where int is the type of the values in the tuple
    """
    
    # Dictionary used to convert argument values to the correct type
    types = {"boolean":ast.literal_eval, "bool":ast.literal_eval, "integer":atoi, "int":atoi,\
             "float":atof, "string":str, "str":str}
    
    result_dict = {}
    
    # Fill up the dictionary with the keyword arguments
    for k, v in dict_kwargs.items():
        # Split the value and type
        split = v.split("|")

        try:
            if len(split) == 2:      
                # Handle conversion from string to boolean
                if split[1] in ("boolean", "bool"):
                    split[0] = split[0].capitalize()
                
                # Convert the value based on the correct type
                result_dict[k] = types[split[1]](split[0])
            
            elif split[1] == "dict":
                # If the argument is a dictionary convert keys and values according to the correct types
                items = split[0].split(";")
                d = {}
                
                for i in items:
                    a,b = i.split(":")
                    
                    # Handle conversion from string to boolean
                    if split[2] in ("boolean", "bool"):
                        a = a.capitalize()
                    if split[3] in ("boolean", "bool"):
                        b = b.capitalize()
                    
                    # Handle None as an item in the dictionary
                    if b == "None":
                        d[types[split[2]](a)] = None
                    else:
                        d[types[split[2]](a)] = types[split[3]](b)
                
                result_dict[k] = d
            
            elif split[1] in ("list", "array", "tuple"):
                # If the argument is a list, array or tuple convert keys and values according to the correct types
                items = split[0].split(";")
                l = []
                
                for i in items:
                    # Handle conversion from string to boolean
                    if split[2] in ("boolean", "bool"):
                        i = i.capitalize()

                    # Handle None as an item in the dictionary
                    if i == "None":
                        l.append(None)
                    else:
                        l.append(types[split[2]](i))
                
                if split[1] == "array":
                    l = np.array(l)
                elif split[1] == "tuple":
                    l = tuple(l)
                    
                result_dict[k] = l
        except IndexError:
            err = "List index out of range. This is most likely due to incorrect syntax of keyword arguments."
            raise Exception(err)
    
    return result_dict

def get_args_by_type(str_args):
    """
    Take in a string of positional arguments and types and convert them to a list of values of the correct type.
    The string should be in the form: 'value1|type1,value2|type2'.
    e.g. '8|int, 0.5|float' would return [8, 0.5]
    """
    
    # Dictionary used to convert argument values to the correct type
    types = {"boolean":ast.literal_eval, "bool":ast.literal_eval, "integer":atoi, "int":atoi,\
             "float":atof, "string":str, "str":str}
    
    result_list = []
    
    # If the argument string is empty return an empty list
    if len(str_args) == 0:
        return list()
    
    for arg in str_args.split(","):
        # Split the value and type
        split = arg.strip().split("|")
        
        try:
            if len(split) == 2:      
                # Handle conversion from string to boolean
                if split[1] in ("boolean", "bool"):
                    split[0] = split[0].capitalize()
                
                # Convert the value based on the correct type
                result_list.append(types[split[1]](split[0]))
            else:
                err = "Incorrect syntax for argument: '{}'. Expected 'value1|type1, value2|type2, ...'".format(str_args)
                raise Exception(err)    
        
        except IndexError:
            err = "List index out of range. This is most likely due to incorrect syntax of arguments."
            raise Exception(err)
        
    return result_list

def convert_types(n_samples, features_df, sort=True):
    """
    Convert data in n_samples to the correct data type based on the feature definitions.
    Optionally sort the data by the feature with "variable_type" set to "identifier". 
    Both n_samples and features_df must be supplied as dataframes. The columns in n_samples must be equal to rows in features_df.
    The features_df dataframe must have "name", "data_type" and "variable_type" columns.
    Accepted data_types are int, float, str, bool.
    """
    
    # Transpose the features dataframe and keep the data_types for each feature
    features_df_t = features_df.T
    features_df_t.columns = features_df_t.loc["name",:].tolist()
    dtypes = features_df_t.loc["data_type",:]
    
    # Dictionary used to convert argument values to the correct type
    types = {"boolean":ast.literal_eval, "bool":ast.literal_eval, "integer":atoi, "int":atoi,\
             "float":atof, "string":str, "str":str}
    
    # Convert columns by the corresponding data type
    for col in n_samples.columns:
        # Handle conversion from string to boolean
        if dtypes[col] in ("boolean", "bool"):
            n_samples.loc[:, col] = n_samples.loc[:, col].astype("str").apply(str.capitalize)
        
        # Convert this column to the correct type
        n_samples.loc[:, col] = n_samples.loc[:, col].apply(types[dtypes[col]])     

    # Get the unique identifier from the feature definitions
    identifier = features_df.loc[features_df["variable_type"] == "identifier"]

    # Ensure the data is sorted by the identifier if available
    if sort and len(identifier) == 1:
        n_samples = n_samples.sort_values(by=[identifier.index[0]], ascending=True)

    return n_samples

def atoi(a):
    """
    Convert a string to float.
    The string can be in the following valid regional number formats:
    4,294,967,295   4 294 967 295   4.294.967.295   4 294 967.295  
    """
    if len(a) == 0:
        return np.NaN
    
    translator = str.maketrans("", "", ",. ")
    
    return (int(a.translate(translator)))

def atof(a):
    """
    Convert a string to float.
    The string can be in the following valid regional number formats:
    4,294,967,295.00   4 294 967 295,000   4.294.967.295,000  
    """
    if len(a) == 0:
        return np.NaN
    
    del_chars = " "
    
    if a.count(",") > 1 or a.rfind(",") < a.rfind("."):
        del_chars = del_chars + ","
    
    if a.count(".") > 1 or a.rfind(",") > a.rfind("."):
        del_chars = del_chars + "."
    
    s = a.translate(str.maketrans("", "", del_chars))

    return float(s.replace(",", "."))

def dict_to_sse_arg(d):
    """
    Converts a dictionary to the argument syntax for this SSE
    """

    # Dictionary used to convert argument values to the correct type
    types = {bool:"bool", int:"int", float:"float", str:"str"}

    # Output string
    s = ""

    # Format into the required syntax
    for k, v in d.items():
        if v is None:
            s = s + str(k) + "=None, "
        else:
            s = s + str(k) + "=" + str(v) + "|" + types[type(v)] + ", "
    
    return s[:-2]

def add_lags(df, lag=1, extrapolate=1, dropna=True, suffix="t"):
    """
    Take in a 2D DataFrame (n_samples by n_features) and create a new DataFrame with lag observations added to it.
    E.g. If lags=2, the previous two observations will be concatenated as inputs to each sample.
    The extrapolate argument can be used to add current (=1) and future observations (>1).
    Samples without enough lags can be retained or dropped using the dropna argument.
    The input DataFrame is assumed to be pre-sorted.
    """
    
    n_features = df.shape[1]
    
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(lag, 0, -1):
        cols.append(df.shift(i))
        names += ["{0}({1}-{2})".format(df.columns[j], suffix, i) for j in range(n_features)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, extrapolate):
        cols.append(df.shift(-i))
        if i == 0:
            names += ["{0}({1})".format(df.columns[j], suffix) for j in range(n_features)]
        else:
            names += ["{0}({1}+{2})".format(df.columns[j], suffix, i) for j in range(n_features)]
    
    # Concatenate the shifted DataFrames
    agg = pd.concat(cols, axis=1)
    # Add the concatenated column names
    agg.columns = names
    
    # drop rows with NaN values
    if dropna:
        agg.dropna(inplace=True)
    
    return agg