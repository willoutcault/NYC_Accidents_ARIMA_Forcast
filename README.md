# Forecasting NYC Traffic Accidents

## Using Autoregressive Integrated Moving Average

### Summary
The data used for this project was retrieved from:
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95

In summary, I wanted to visualize the impact COVID-19 had on the total number of NYC traffic accidents. This involved forecasting future months based off previous data not influenced by the pandemic.

### ARIMA models

To do the forecasting I created a Autoregressive Integrated Moving Average model. This model can be broken into three parts. 

First Autoregression (AR) refers to a model that shows a changing variable that regresses on its own lagged, or prior, values.

Next Integrated (I) represents the differencing of raw observations to allow for the time series to become stationary. 

Lastly the Moving Average (MA) incorporates the dependency between an observation and a residual error from a moving average model applied to lagged observations.
