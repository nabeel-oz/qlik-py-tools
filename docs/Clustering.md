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
| return | The output of the expression | `labels`, `probabilities` | `labels` refers to the clustering classification. This is the default value if the parameter is not specified. <br/><br/>The HDBSCAN library implements soft clustering, where each data point is assigned a cluster membership score ranging from 0.0 to 1.0. A score of 0.0 represents a sample that is not in the cluster at all (all noise points will get this score) while a score of 1.0 represents a sample that is at the heart of the cluster (note that this is not the spatial centroid notion of core). You can access these scores via the argument `return=probabilities`. |
| debug | Flag to output additional information to the terminal and logs | `true`, `false` | Information will be printed to the terminal as well to a log file: `..\qlik-py-env\core\logs\Cluster Log <n>.txt`. Particularly useful is looking at the input and output Data Frames. |
| load_script | Flag to set the output format for the function | `true`, `false` | Set to `true` if calling the functions from the load script in the Qlik app. This will change the output to a table consisting of two fields the `key` which is the first dimension being clustered, and the specified return value (`labels` or `probabilities`). <br/><br/>You do not need to specify this parameter for the `Cluster_by_Dim` function as that can only be used through the load script. |
| missing | Strategy for handling missing / null values | `zeros`, `mean`, `median`, `mode` | Any missing values in the data need to be handled before the clustering algorithm can be executed. The strategy will depend on the type of data. The default value is `zeros` |
| scaler | Strategy for standardizing the data so that certain features don't skew the results | `standard`, `minmax`, `maxabs`, `robust`, `quantile`, `none` | Standardizing the data is a common requirement for machine learning algorithmns. In this implementation we use the [sklearn.preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html) package |

### Scaler Parameters

### HDBSCAN Parameters

## Use Clustering with your own app

You should have completed the installation instructions in the master README.md.

The [sample app](Sample_App_Clustering.qvf) can be used as a template for the instructions below.

## Attribution
The data used in the sample app was obtained from https://www.data.vic.gov.au/:
- [Crime Statistics Agency Data Tables](https://www.data.vic.gov.au/data/dataset/crime-by-location-data-table) 
- [Crash Stats Data Extract](https://www.data.vic.gov.au/data/dataset/crash-stats-data-extract)
