# Time series forecasting with Facebook Prophet

Forecasting is something that can add value to almost any Qlik Sense app that contains time series data. Many executives these days are looking to the future, even if they are still stuck using tabular reports! The power of modern algorithms provided through Qlik's associative model can be a great leap forward.

Last year Facebook's data science team released an open source forecasting tool for Python and R. There is no shortage of forecasting algorithms out there, but this one is impressive in how effortlessly it produces high quality forecasts. I recommend a quick read of [Facebook's post](https://research.fb.com/prophet-forecasting-at-scale/).

While this is a great tool, there is a scarcity of people with the skills to use such tools at most organizations. Qlik's advanced analytics integration can bridge this gap by providing a simple, interactive experience for generating quality forecasts.

The Prophet functions in this SSE provides almost all of Prophet's capabilities in Qlik through simple expressions.

## Quick Start

You use the Prophet functions in Qlik with the syntax:

`<Analytic connection name>.Prophet([Date Column], [Value Column], 'arg1=value1, arg2=value2, ...')`

There are a few variants of the Prophet function made available, but this is the main function and we'll look at the other ones later.

Here's an example of an actual expression:

`PyTools.Prophet(FORECAST_MONTH, Count({$<FORECAST_LINK_TYPE = {'Actual'}>} Distinct ACCIDENT_NO), 'freq=MS, return=yhat')`

In this example the first column is the forecast month, the second column is the measure we want to forecast and the third column is a string of additional key word arguments.

Using this expression in a line chart, with the dimension as forecast month, and another measure to show the actual values, would let us observe the actual and forecast values.

## Additional Parameters

The additional arguments provided through the last parameter let you use the different features of Prophet.

Any of these arguments can be included in the final string parameter for the Prophet function using the syntax: 'argument=value'. Separate arguments with a comma and use single quotes around the entire string.

| Key word | Description | Sample Values | Remarks |
| --- | --- | --- | --- |
| return | The output of the expression | `yhat`, `yhat_upper`, `yhat_lower`, `y_then_yhat`, `y_then_yhat_upper`, `y_then_yhat_lower`, `trend`, `trend_upper`, `trend_lower`, `seasonal`, `seasonal_upper`, `seasonal_lower`, `yearly`, `yearly_upper`, `yearly_lower` & any other column in the forecast output | yhat refers to the forecast values. This is the default value. The y_then_yhat options allow you to plot the actual values for historical data and forecast values only for future dates. Upper and lower limits are available for each type of output. |
| freq | The frequency of the time series | `D`, `MS`, `M`, `H`, `T`, `S`, `ms`, `us` | The most common options would be D for Daily, MS for Month Start and M for Month End. The default value is D, however this will mess up results if you provide the values in a different frequency, so always specify the frequency. See the full set of options [here](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases). |
| debug | Flag to output additional information to the terminal and logs | `true`, `false` | Information will be printed to the terminal as well to a log file: `..\qlik-py-env\core\logs\Prophet Log <n>.txt`. Particularly useful is looking at the Request Data Frame to see what you are sending to the algorithm and the Forecast Data Frame to see the possible result columns. |
| take_log | Take a logarithm of the values before forecasting | `true`, `false` | Default value is `false`. This can be applied when making the time series more stationary might improve forecast values. You can just try both options and compare the results. In either case the values are returned in the original scale. |
| cap | A saturating maximum for the forecast | A decimal or integer value e.g. `1000000` | You can apply a logistic growth trend model using this argument. For example when the maximum market size is known. More information [here](https://facebook.github.io/prophet/docs/saturating_forecasts.html). |
| floor | A saturating minimum for the forecast | A decimal or integer value e.g. `0` | This argument must be used in combination with a cap. |
| changepoint_prior_scale | A parameter to adjust the trend flexibility | A decimal value e.g. `0.05` | If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), you can try adjusting this parameter. The default value is `0.05`. Increasing it will make the trend more flexible. Decreasing it will make the trend less flexible. More information [here](https://facebook.github.io/prophet/docs/trend_changepoints.html). |
| interval_width | The width of the uncertainty intervals | A decimal value e.g. `0.8` | The default value is `0.8` (80%). More information [here](https://facebook.github.io/prophet/docs/uncertainty_intervals.html). |
| add_seasonality | Additional seasonality to be considered in the forecast. | A string value which represents the name of the seasonality e.g. `monthly` | Prophet will by default fit weekly and yearly seasonalities, if the time series is more than two cycles long. It will also fit daily seasonality for a sub-daily time series. You can add other seasonalities (monthly, quarterly, hourly) using this parameter. More information [here](https://facebook.github.io/prophet/docs/seasonality_and_holiday_effects.html). |
| seasonality_period | Period for the additional seasonality | A decimal or integer value e.g. `30.5` | This is the period of the seasonality in days. |
| seasonality_fourier | Fourier terms for the additional seasonality | An integer value e.g. `5` | For reference, by default Prophet uses 3 terms for weekly seasonality and 10 for yearly seasonality. Increasing the number of Fourier terms allows the seasonality to fit faster changing cycles, but can also lead to overfitting. |
| seasonality_prior_scale | The extent to which the seasonality model will fit the data | A decimal or integer value e.g. `0.05` | If you find that the seasonalities are overfitting, you can adjust the prior scale to smooth them using this parameter. |
| holidays_prior_scale | The magnitude of the holiday effect, if holidays are included in the function | A decimal or integer value e.g. `10` | If you find that the holidays are overfitting, you can adjust their prior scale to smooth them using this parameter. By default this parameter is `10`, which provides very little regularization. Reducing this parameter dampens holiday effects. |
| weekly_start | Set the start of the week when calculating weekly seasonality | An integer value e.g. `6` (for Monday) | Only relevant when the using the Prophet_Seasonality function to get the weekly seasonality. See more below in the Seasonality section. `0` represents Sunday, `6` represents Monday. |
| yearly_start | Set the start of the year when calculating yearly seasonality | An integer value e.g. `0` (for 1st Jan) | Only relevant when the using the Prophet_Seasonality function to get the yearly seasonality. See more below in the Seasonality section. `0` represents 1st Jan, `1` represents 2nd Jan and so on. |
| lower_window | Extend the holidays by certain no. of days prior to the date. | A negative integer value e.g. `-1` | Only relevant when passing holidays to Prophet. This can be used to analyze holiday effects before a holiday e.g. 7 days before Christmas. |
| upper_window | Extend the holidays by certain no. of days after the date. | A positive integer value e.g. `1` | Only relevant when passing holidays to Prophet. This can be used to analyze holiday effects after a holiday e.g. 1 day after New Year. |

