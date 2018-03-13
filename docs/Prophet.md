# Time series forecasting with Facebook Prophet

Forecasting is something that can add value to almost any Qlik Sense app that contains time series data. Many executives these days are looking to the future, even if they are still stuck using tabular reports! The power of modern algorithms provided through Qlik's associative model can be a great leap forward.

Last year Facebook's data science team released an open source forecasting tool for Python and R. There is no shortage of forecasting algorithms out there, but this one is impressive in how effortlessly it produces high quality forecasts. I recommend a quick read of [Facebook's post](https://research.fb.com/prophet-forecasting-at-scale/).

While this is a great tool, there is a scarcity of data scientists at most organizations. Qlik's advanced analytics integration can bridge this gap by providing a simple, interactive experience to generating quality forecasts.

The Prophet functions in this SSE provides almost all of Prophet's capabilities in Qlik through simple expressions.

## Quick Start

You use the Prophet functions in Qlik with the syntax:

`<Analytic connection name>.Prophet([Date Column], [Value Column], 'arg1=value1, arg2=value2, ...')`

There are a few variants of the Prophet function made available, but this is the main function and we'll look at the other ones later.

Here's an example of an actual expression:

`PyTools.Prophet(FORECAST_MONTH, Count({$<FORECAST_LINK_TYPE = {'Actual'}>} Distinct ACCIDENT_NO), 'freq=MS, return=yhat')`

In this example the first column is the forecast month, the second column is the measure we want to forecast and the third column is a string of additional key word arguments.

Using this expression in a line chart, with the dimension as forecast month, and another measure to show the actual values, would let us observe the actual and forecast values.