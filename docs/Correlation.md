# Correlations

The correlation methods in this Server Side Extension (SSE) take in two columns of data and compute the linear correlation. 

The methods are implemented using Pandas, specifically the [pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) method.

## Why not just use the Qlik "Correl" function?

The Correl funtion is the simpler approach and will generally give you better performance. However, in certain cases this function may not be flexible enough. For example, if you have several series in your data and want to dynamically select one and calculate the correlations versus the others, you will find the out-of-the-box function is insufficient.

In addition, this SSE allows you to calculate three types of correlations: Pearson, Spearman Rank and Kendall Tau. 

## Quick Start

There are several methods for calculating correlation, but the most commonly used statistic is the Pearson r correlation coefficient. This can be calculated using the Pearson function with the syntax:

`<Analytic connection name>.Pearson([Series 1 as string], [Series 2 as string])`

The two series are passed an a concatenated string of values, which gives us flexibility in using the result against various dimensions and visualizations. The string should take the form of semi-colon separated values for example:

`PyTools.Pearson('10;12;14', '1;2;3')`

You can also use the Correlation function which takes in a third argument to specify the correlation type:

`<Analytic connection name>.Correlation([Series 1 as string], [Series 2 as string], 'Correlation Type')`

Possible values for the Correlation Type parameter are: 
- pearson
- spearman
- kendall

## Use Correlations in your own app

While the Correlation function is straight forward, you will need to set up a few expressions in Qlik to use it dynamically against multiple data series.

The [sample app](Sample_App_Correlations.qvf) can be used as a template for the instructions below.

In this app we have 194 different Indicators, with values and rankings for each Local Government Area (LGA) in Victoria, Australia. There are 79 LGAs, so each Indicator has a series of 79 values. 

Here's an expression to calculate the correlation of one specific Indicator versus all the others:

`PyTools.Correlation($(vSeriesInd1), $(vSeriesExcInd1), 'pearson')`

Plotted against the "Indicator" dimension, we will get a correlation coefficient for each row.

The first variable in the expression remains constant for all Indicators, and is based on a selection. The second variable gives us a series for each row in the visualization, but excludes the series in the first variable. This is perhaps better understood by looking at the screenshot below:

![Steps to get to the correlation](images/Correlations-01.png)

Here are the expressions for the two variables:

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

The `vIndicator1` variable in `vSeriesInd1` is fixing the 
