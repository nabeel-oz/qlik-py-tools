# Clustering with HDBSCAN

Clustering is an unsupervised machine learning technique that involves grouping of data points based on the similarity of relevant features. There are several clustering algorithms avaialble, but I've picked [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html) for this implementation as it gives good quality results for exploratory data analysis and performs really well. A good comparison of several clustering algorithms in Python is covered in the [HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html). 

The Cluster functions in this SSE provides the capabilities of HDBSCAN in Qlik through simple expressions.

## Quick Start

### Clustering with multiple features

The `Cluster` function takes in three parameters: the dimension to be clustered, the set of features as a concatenated string, and a string where we can supply optional arguments.

```
<Analytic connection name>.Cluster([Dimension], [Measure 1] & ';' & [Meaure n], 'arg1=value1, arg2=value2, ...')
```

The features need to be a semi-colon separated string. Optional parameters are covered in more detail in the Additional Parameters section below.

Here's an example of an actual expression:

```
PyTools.Cluster([Local Government Area], sum([Incidents Recorded]) & ';' & avg(ERP), 'return=labels,scaler=robust,min_cluster_size=3')
```

### Clustering by a second dimension

### Geospatial clustering

## Additional Parameters

| Keyword | Description | Sample Values | Remarks |
| --- | --- | --- | --- |
| return | The output of the expression | `labels`, `probabilities` | `labels` refers to the clustering classification. This is the default value. |
| debug | Flag to output additional information to the terminal and logs | `true`, `false` | Information will be printed to the terminal as well to a log file: `..\qlik-py-env\core\logs\Cluster Log <n>.txt`. Particularly useful is looking at the input and output Data Frames. |

## Use Clustering with your own app

You should have completed the installation instructions in the master README.md.

The [sample app](Sample_App_Clustering.qvf) can be used as a template for the instructions below.

## Attribution
The data used in the sample app was obtained from https://www.data.vic.gov.au/:
- [Crime Statistics Agency Data Tables](https://www.data.vic.gov.au/data/dataset/crime-by-location-data-table) 
- [Crash Stats Data Extract](https://www.data.vic.gov.au/data/dataset/crash-stats-data-extract)
