'''

Code for the dash app

docs:
https://dash-bootstrap-components.opensource.faculty.ai/docs/
https://dash.plotly.com/dash-html-components
7

'''

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from plotly.subplots import make_subplots

import plotly.graph_objects as go
import numpy as np
import pandas as pd

from graphs import make_plot


app = dash.Dash(__name__, title = 'Water scarcity', external_stylesheets = [dbc.themes.BOOTSTRAP])


server = app.server





