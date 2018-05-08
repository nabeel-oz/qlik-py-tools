# Clustering with HDBSCAN

Clustering is an unsupervised machine learning technique that involves grouping of data points based on the similarity of relevant features. There are several clustering algorithms avaialble, but I've picked [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html) for this implementation as it gives good quality results for exploratory data analysis and performs really well. A good comparison of several clustering algorithms in Python is covered in the [HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html). 

The Cluster functions in this SSE provides the capabilities of HDBSCAN in Qlik through simple expressions.

## Quick Start

The HDBSCAN algorithm will classify the input dimension into clusters. The labels are the default output, with `-1` representing outliers and labels `0` and above representing clusters.

### Clustering with multiple features

The `Cluster` function takes in three parameters: the dimension to be clustered, the set of features as a concatenated string, and a string where we can supply optional key word arguments.

```
<Analytic connection name>.Cluster([Dimension], [Measure 1] & ';' & [Meaure 2] ..., 'arg1=value1, arg2=value2, ...')
```

The features need to be a semi-colon separated string. Optional key word arguments are covered in more detail in the Additional Parameters section below.

Here's an example of an actual expression:

```
PyTools.Cluster([Local Government Area], sum([Incidents Recorded]) & ';' & avg(ERP), 'scaler=robust, min_cluster_size=3')
```

In this example we are clustering Local Government Areas in Victoria, Australia by the number of crimes and the estimated residential population. We are also using the key word arguments to specify that we want to apply scaling to our features using the scikit-learn RobustScaler and we want the minimum cluster to be at least 3 LGAs.

### Clustering by a second dimension

If clusters are being classified as data is loaded into Qlik we can use a second dimension to create features based on an expression. The function will pivot the data using this second dimension, before scanning for clusters.

This is done using the `Cluster_by_Dim` function:

```
LOAD
    key,
    labels
EXTENSION <Analytic connection name>.Cluster_by_Dim(TableName{[Dimension 1], [Dimension 2], [Expression], 'arg1=value1, arg2=value2, ...'})
```

This function can only be used in the Qlik Load Script as the [input and output number of rows will be different](https://github.com/qlik-oss/server-side-extension/blob/master/docs/limitations.md#expressions-using-sse-must-persist-the-cardinality).

Here's an example where we use the `Cluster_by_Dim` function using the [LOAD...EXTENSION](https://help.qlik.com/en-US/sense/April2018/Subsystems/Hub/Content/Scripting/ScriptRegularStatements/Load.htm) syntax:

```
[LGA Clusters by Subgroup]:
LOAD
    key as [Local Government Area],
    labels as [Clusters by Subgroup]
EXTENSION PyTools.Cluster_by_Dim(TempInputsTwoDims{LGA, SubGroup, Rate, Args});
```

The function returns a table with two fields; the first dimension in the input parameters and the corresponding clustering labels.

### Geospatial clustering

The HDBSCAN algorithm works well with geospatial coordinates as well. For this you can use the `Cluster_Geo` function:

```
<Analytic connection name>.Cluster_Geo([Dimension], [Latitude], [Longitude], 'arg1=value1, arg2=value2, ...')
```

The latitude and longitude need to be provided as separate fields in decimal format. 

Here's an example where we cluster accidents by location:

```
PyTools.Cluster_Geo(ACCIDENT_NO, Lat, Long, '')
```

![geospatial clustering](images/Clustering-01.png)

## Additional Parameters

The additional arguments provided through the last parameter in the functions give you control over the clustering output. If you don't need to specify any additional arguments you can pass an empty string.

Any of these arguments below can be included in the final string parameter for the clustering functions defined above using the syntax: `argument=value`. Separate arguments with a comma and use single quotes around the entire string.

### Basic Parameters

| Keyword | Description | Sample Values | Remarks |
| --- | --- | --- | --- |
| return | The output of the expression. | `labels`, `probabilities` | `labels` refers to the clustering classification. This is the default value if the parameter is not specified. The cluster labels start at 0 and count up. HDBSCAN is noise aware and has a notion of data samples that are not assigned to any cluster. This is handled by assigning these samples the label -1. <br/><br/>The HDBSCAN library implements soft clustering, where each data point is assigned a cluster membership score ranging from 0.0 to 1.0. You can access these scores via the argument `return=probabilities`. |
| debug | Flag to output additional information to the terminal and logs. | `true`, `false` | Information will be printed to the terminal as well to a log file: `..\qlik-py-env\core\logs\Cluster Log <n>.txt`. Particularly useful is looking at the input and output Data Frames. <br/><br/>The default value is `false`. |
| load_script | Flag to set the output format for the function. | `true`, `false` | Set to `true` if calling the functions from the load script in the Qlik app. This will change the output to a table consisting of two fields the `key` which is the first dimension being clustered, and the specified return value (`labels` or `probabilities`). <br/><br/>You do not need to specify this parameter for the `Cluster_by_Dim` function as that can only be used through the load script. The default value for the `Cluster` and `Cluster_Geo` function is `false`. |
| missing | Strategy for handling missing / null values. | `zeros`, `mean`, `median`, `mode` | Any missing values in the data need to be handled before the clustering algorithm can be executed. The strategy will depend on the type of data. The default value is `zeros`. |
| scaler | Strategy for standardizing the data so that certain features don't skew the results. | `standard`, `minmax`, `maxabs`, `robust`, `quantile`, `none` | Standardizing the data is a common requirement for machine learning algorithmns. In this implementation we use the [sklearn.preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html) package. <br/><br/>The default value is `robust`. |

### Scaler Parameters

The scaling options provided in this implementation make use of the scikit-learn library. For a better understanding of the options please refer to the documentation [here](http://scikit-learn.org/stable/modules/preprocessing.html).

| Keyword | Description | Sample Values | Remarks |
| --- | --- | --- | --- |
| with_mean | An option for the [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler). <br/><br/>If `true`, center the data before scaling. | `true`, `false` | The default value is `true`. |
| with_std | An option for the [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler). <br/><br/>If `true`, scale the data to unit variance (or equivalently, unit standard deviation). | `true`, `false` | The default value is `true`. |
| feature_range | An option for the [MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler). <br/><br/>Desired range of transformed data. | `(0;100)` | The default value is `(0;1)`. |
| with_centering | An option for the [RobustScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler). <br/><br/>If `true`, center the data before scaling. | `true`, `false` | The default value is `true`. |
| with_scaling | An option for the [RobustScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler). <br/><br/>If `true`, scale the data to interquartile range. | `true`, `false` | The default value is `true`. |
| quantile_range | An option for the [RobustScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler). <br/><br/>Quantile range used to calculate the scale. | `(10.0;90.0)` | The default value is `(25.0;75.0)`. |
| n_quantiles | An option for the [QuantileTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer). <br/><br/>Number of quantiles to be computed. It corresponds to the number of landmarks used to discretize the cumulative density function. | `1000` | The default value is `1000`. |
| output_distribution | An option for the [QuantileTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer). <br/><br/>Marginal distribution for the transformed data. | `uniform`, `normal` | The default value is `uniform`. |
| ignore_implicit_zeros | An option for the [QuantileTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer). <br/><br/>Only applies to sparse matrices. If True, the sparse entries of the matrix are discarded to compute the quantile statistics. If False, these entries are treated as zeros. | `true`, `false` | The default value is `false`. |
| subsample | An option for the [QuantileTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer). <br/><br/>Maximum number of samples used to estimate the quantiles for computational efficiency. | `100000` | The default value is `100000`. |
| random_state | An option for the [QuantileTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer). <br/><br/>If int, random_state is the seed used by the random number generator. If this is not specified, the random number generator is the RandomState instance used by np.random. Note that this is used by subsampling and smoothing noise. | `1` | The default value is None. |

### HDBSCAN Parameters

Most of the options available for the HDBSCAN class documented in the [API Reference](https://hdbscan.readthedocs.io/en/latest/api.html) are included in this implementation.

| Keyword | Description | Sample Values | Remarks |
| --- | --- | --- | --- |
| algorithm | Exactly which algorithm to use; HDBSCAN has variants specialised for different characteristics of the data. By default this is set to `best` which chooses the “best” algorithm given the nature of the data. You can force other options if you believe you know better. | `best`, `generic`, `prims_kdtree`, `prims_balltree`, `boruvka_kdtree`, `boruvka_balltree` | The default value is `best`. |
| metric | The metric to use when calculating distance between instances in a feature array.  | `euclidean`, `manhattan`, `canberra`, `precomputed` | A large number of distance metrics are avaialble. For a full list see [here](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html#what-about-different-metrics). For a better understanding of distance metrics refer to the [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html). <br/><br/>The default value is `euclidean`. For the `Cluster_Geo` function the default value is `haversine`. |
| min_cluster_size | The minimum size of clusters. | `3` | More information [here](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size). <br/><br/>The default value is `5`. |
| min_samples | The number of samples in a neighbourhood for a point to be considered a core point. | `5` | More information [here](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-samples). <br/><br/>The default value is None. |
| p | p value to use if using the minkowski metric. | `2` | The default value is None. |
| alpha | A distance scaling parameter as used in robust single linkage. | `1.0` | In practice it is best not to mess with this parameter. More information [here](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-alpha). <br/><br/>The default value is `1.0`. |
| cluster_selection_method | The method used to select clusters from the condensed tree. | `eom`, `leaf` | If you are more interested in having small homogeneous clusters then you may find the default option, Excess of Mass, has a tendency to pick one or two large clusters and then a number of small extra clusters. You can use the `leaf` option to select leaf nodes from the tree, producing many small homogeneous clusters. Note that you can still get variable density clusters via this method, and it is also still possible to get large clusters, but there will be a tendency to produce a more fine grained clustering than Excess of Mass can provide. More information [here](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering). <br/><br/>The default value is `eom`. |
| allow_single_cluster | By default HDBSCAN* will not produce a single cluster. Setting this to `true` will override this and allow single cluster results. | `true`, `false` | More information [here](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#allowing-a-single-cluster). <br/><br/>The default value is `false`. |
| match_reference_implementation | Flag to switch between the standard HDBSCAN implementation and the original authors reference implementation in Java. | `true`, `false` | This can result in very minor differences in clustering results. Setting this flag to `true` will, at a some performance cost, ensure that the clustering results match the reference implementation. <br/><br/>The default value is `false`. |

## Use Clustering with your own app

You should have completed the installation instructions in the master README.md.

The [sample app](Sample_App_Clustering.qvf) can be used as a template for the instructions below.

## Attribution
The data used in the sample app was obtained from https://www.data.vic.gov.au/:
- [Crime Statistics Agency Data Tables](https://www.data.vic.gov.au/data/dataset/crime-by-location-data-table) 
- [Crash Stats Data Extract](https://www.data.vic.gov.au/data/dataset/crash-stats-data-extract)
