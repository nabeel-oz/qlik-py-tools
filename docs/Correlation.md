# Correlations

The correlation methods in this Server Side Extension (SSE) take in two columns of data and compute the linear correlation. 

The methods are implemented using Pandas, specifically the [pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) method.

## Why not just use the Qlik "Correl" function?

The Correl funtion is the simpler approach and will generally give you better performance. However, in certain cases this function may not be flexible enough. For example, if you have several series in your data and want to dynamically select one and calculate the correlations versus the others, you will find the out-of-the-box function is insufficient.

In addition, this SSE allows you to calculate three types of correlations: Pearson, Spearman Rank and Kendall Tau. 

## Quick Start

The most commonly used linear correlatinon is the Pearson r correlation coefficient. This can be calculated using the Pearson function with the syntax:

`<Analytic connection name>.Pearson([Series 1 as string], [Series 2 as string])`

The two series are passed an a concatenated string of values, which gives us flexibility in using the result against various dimensions and visualizations. The string should take the form of comma separated values for example:

`PyTools.Pearson('10;12;14', '1;2;3')`

You can also use the Correlation function which takes in a third argument to specify the correlation type:

`<Analytic connection name>.Correlation([Series 1 as string], [Series 2 as string], 'Correlation Type')`

Possible values for the Correlation Type parameter are: 
- pearson
- spearman
- kendall

## Use Correlations in your own app

Here's an example of an actual expression:

`PyTools.Pearson($(vSeriesInd1), $(vSeriesExcInd1))`

The variables in this expression are concatenated strings of values. The first variable remains constant for all rows in the visualization, while the second variable excludes the series in the first variable. 

Plotted against the "Indicator" dimension, we will get a correlation coefficient for each indicator versus the constant indicator in the first series.

```
//vSeriesInd1
Keepchar(
	Concat(TOTAL
        Aggr(
            Only({$<[Profile Type] = {'LGA'}, [Indicator] = {"$(vIndicator1)"}>} ([LGA Name] & ':' & Value)), 
            (Indicator, (TEXT, ASCENDING)), ([LGA Name], (TEXT, ASCENDING))
            ) 
        , '; ')
	, '0123456789.;')
```

```
//vSeriesExcInd1
Keepchar(
	Concat({$<[Indicator] = {*}>}
        Aggr(
            Only({$<[Profile Type] = {'LGA'}, [Correlation Relevant] = {'Yes'}, [Indicator]={*}-{"$(vIndicator1)"}>} ([LGA Name] & ':' & Value)), 
            (Indicator, (TEXT, ASCENDING)), ([LGA Name], (TEXT, ASCENDING))
            ) 
        , '; ')
    , '0123456789.;')
```
