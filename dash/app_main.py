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


# Plots

fig1 = make_subplots(rows = 1, cols = 1)


### INPUT SELECTIONS ###

# Target variable
drop_trgt = dcc.Dropdown(
    id = 'id_target_var',
    options = [{"label": 'Water Scarcity', 'value': 'WS'},
        {"label": 'Water USe Efficiency', 'value': 'WUE'}],
    value = 'WS')

# Country selection
drop_country = dcc.Dropdown(
    id = 'id_sel_country',
    options = [], ### Need to create dictionary with full names of countries and their corresponding country codes
    
    value = 'BEL')

# Climate prediciton selection
drop_climate = dcc.Dropdown(
    id = 'id_sel_climate',
    options = [{"label": 'Pessimistic', 'value': ''},
        {"label": 'Neutral', 'value': ''},
        {"label": 'Optimistic', 'value': ''}],
    value = '')

# Input group for the changes in 
ins_changes = dbc.Row(dbc.Col(
    html.Div([
    dbc.InputGroup([
        dbc.InputGroupAddon("Population change", addon_type="prepend"),
        dbc.Input(id='ch_pop', type="number", value=1.1, min=-10, max=10, step=0.1)
        ]),
    dbc.InputGroup([
        dbc.InputGroupAddon("Urbanisation", addon_type="prepend"),
        dbc.Input(id='ch_urban', type="number", value=1.5, min=-5, max=5, step=0.1)
        ]),
    dbc.InputGroup([
        dbc.InputGroupAddon("GDP change", addon_type="preprend"),
        dbc.Input(id='ch_gdp', type="number", value=2, min=-10, max=10, step=0.1)
        ]),
    dbc.InputGroup([
        dbc.InputGroupAddon("Mortality rate change", addon_type="prepend"),
        dbc.Input(id='ch_mort', type="number", value=0, min=-5, max=5, step=0.1)
        ]),
    dropdown])
))


app.layout = dbc.Container([
    html.Div(
        children=[
            html.H1(children='Water scarcity'),
            html.H2(children='Predict the water scarcity in your country over the next years.')
            ]
        ),
    html.Hr(),
    dbc.Row([])
])





@app.callback(
    Output(),
    [Input('id_target_var', 'value'),
    Input('id_sel_country', 'value'),
    Input('id_sel_climate', 'value')
    ]
)



def update_plot(target_var, country, climate, )



