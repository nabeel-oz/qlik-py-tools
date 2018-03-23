# Correlations

The correlation methods in this Server Side Extension (SSE) take in two columns of data and compute the linear correlation. 

The methods are implemented using Pandas, specifically the [pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) method.

## Why not just use the Qlik "Correl" function?

The Correl funtion is the simpler approach and will generally give you better performance. However, in certain cases this function may not be flexible enough. For example, if you have several series in your data and want to dynamically select one and calculate the correlations versus the others, you will find the out of the box function may not be sufficient.

In addition, this SSE allows you to calculate three types of correlations: Pearson, Spearman Rank and Kendall Tau. 
