
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

import functions

app = dash.Dash()

indexedDataset = functions.read_data()

indexedDataset_logScale = functions.get_log_df(indexedDataset)

smodel = functions.get_model(indexedDataset_logScale)

functions.plot_year(smodel, indexedDataset_logScale, 24)

functions.plot_forecast(smodel, indexedDataset_logScale, 24)


if __name__ == '__main__':
    app.run_server(debug=True)
