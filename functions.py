import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go

import pandas as pd
import numpy as np
import pmdarima as pm
from IPython.display import display, HTML

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import matplotlib.dates as mdates
import matplotlib as mpl

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from datetime import datetime as dt

import warnings


def read_data():
    df = pd.read_csv("https://raw.githubusercontent.com/willoutcault/NYC_Accidents_ARIMA_Forcast/master/Accidents.csv")
    df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
    df = df.set_index(['Month'])
    warnings.filterwarnings('ignore')
    indexedDataset = df[0:91]
    return(indexedDataset)

def test_stationarity(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()
    movingStd = timeseries.rolling(window=12).std()
    orig = plt.plot(timeseries, color='darkgreen', label='Original')
    mean = plt.plot(movingAverage, color='blue', label='Rolling Mean')
    std = plt.plot(movingStd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['Accidents'], autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def get_log_df(indexedDataset):
    indexedDataset_logScale = np.log(indexedDataset)
    return(indexedDataset_logScale)

def get_model(indexedDataset_logScale):
    smodel = pm.auto_arima(indexedDataset_logScale['Accidents'], start_p=1, start_q=1,
        test='adf',
        max_p=3, max_q=3, m=12,
        start_P=0, seasonal=True,
        d=None, D=1, trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True);
    return(smodel)

def plot_forecast(smodel, indexedDataset_logScale, n_periods):
    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(indexedDataset_logScale.index[-1], periods = n_periods, freq='MS')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = np.exp(pd.Series(confint[:, 0], index=index_of_fc))
    upper_series = np.exp(pd.Series(confint[:, 1], index=index_of_fc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indexedDataset_logScale.iloc[:,0].index,
        y=np.exp(indexedDataset_logScale.iloc[:,0].values),
        mode='lines+markers',
        name='Actual'))
    fig.add_trace(go.Scatter(
        x=fitted_series.index,
        y=np.exp(fitted_series.values),
        mode='lines+markers',
        line_color='rgba(232,68,68,1)',
        name='Forecast'))
    fig.add_trace(go.Scatter(
        x=fitted_series.index,
        y=lower_series,
        fill='tonexty',
        showlegend=False,
        fillcolor='rgba(246,194,194,.5)',
        line_color='rgba(246,194,194,.5)',
        name = "Forecast"))
    fig.add_trace(go.Scatter(
        x=fitted_series.index,
        y=upper_series,
        fill='tonexty',
        showlegend=False,
        fillcolor='rgba(246,194,194,.5)',
        line_color='rgba(246,194,194,.5)',
        name = "Forecast"))
    fig.update_layout(title='2012 - 2019 Auto Accidents with Projections',
                       xaxis_title='Month',
                       yaxis_title='Accidents')
    fig.show()

def plot_year(smodel, indexedDataset_logScale, n_periods):

    df = pd.read_csv("https://raw.githubusercontent.com/willoutcault/NYC_Accidents_ARIMA_Forcast/master/Accidents.csv")
    df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
    df = df.set_index(['Month'])
    warnings.filterwarnings('ignore')

    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(indexedDataset_logScale.index[-1], periods = n_periods, freq='MS')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = np.exp(pd.Series(confint[:, 0], index=index_of_fc))
    upper_series = np.exp(pd.Series(confint[:, 1], index=index_of_fc))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fitted_series[2:].index, y=np.exp(fitted_series[2:].values),
                mode='lines',
                name='Fitted Series',
                line=dict(dash='dot', width=4)))

    fig.add_trace(go.Scatter(x=df.iloc[68:,0].index, y=df.iloc[68:,0].values,
                mode='lines+markers',
                name='Actual'))

    fig.update_layout(title='Current Day Auto Accidents',
               xaxis_title='Month',
               yaxis_title='Accidents')

    fig.show()

def create_table(fitted_series, df):
    fitted_series_scaled = np.exp(fitted_series).round(decimals=0).astype(object)
    perc_change = (fitted_series_scaled.values[0:4]-df.values[90:94,0])/fitted_series_scaled.values[0:4]*(-100)
    dt = pd.DataFrame({"Month":df.index[90:94], "Projections":fitted_series_scaled.values[0:4],
                  "Actual":df.values[90:94,0], "Percent Change":perc_change})
    print(dt)
