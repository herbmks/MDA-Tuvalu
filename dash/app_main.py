'''

This script includes the front-end of the Dash app.

The backend functionality is contained in the tools.py script.

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

import tools


'''
Geeneral set up of the necessary tools for the app to functions.
'''
app = dash.Dash(__name__, title='Water scarcity', external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server
modelbackend = tools.PredModels()


# Plots
fig = make_subplots(rows = 1, cols = 1)

fig.add_trace(
    go.Scatter(x = np.arange(0, 10, 1),
        y = np.arange(80, 90, 1),
        name = 'Test'),
    row=1, col=1)
fig.update_layout(width = 800)


'''
Creating the input fields for the different variables.
'''

drop_trgt = dcc.Dropdown(
    id = 'id_target_var',
    options = [{"label": 'Water Scarcity (MDG)', 'value': 'WS_MDG'},
        {"label": 'Water Use Efficiency', 'value': 'WUE_SDG'},
        {"label": 'Water Scarcity (SDG)', 'value': 'WS_SDG'}],
    placeholder = "Select water scarcity indicator")

drop_country = dcc.Dropdown(
    id = 'id_sel_country',
    options = modelbackend.get_country_dict(), ### Need to create dictionary with full names of countries and their corresponding country codes
    placeholder = "Select country")

drop_climate = dcc.Dropdown(
    id = 'id_sel_climate',
    options = [{"label": 'Pessimistic', 'value': 'high'},
        {"label": 'Neutral', 'value': 'med'},
        {"label": 'Optimistic', 'value': 'low'}],
    placeholder = "Select the climate scenario")

ins_changes = dbc.Row(dbc.Col(
    html.Div([
    drop_trgt,
    drop_country,
    drop_climate,
    dbc.InputGroup([
        dbc.InputGroupAddon("Population", addon_type="prepend"),
        dbc.Input(id='id_ch_pop', type="number", value=1.1, min=-10, max=10, step=0.1)
        ]),
    dbc.InputGroup([
        dbc.InputGroupAddon("Urbanisation", addon_type="prepend"),
        dbc.Input(id='id_ch_urban', type="number", value=1.5, min=-5, max=5, step=0.1)
        ]),
    dbc.InputGroup([
        dbc.InputGroupAddon("GDP per capita", addon_type="preprend"),
        dbc.Input(id='id_ch_gdp', type="number", value=2, min=-10, max=10, step=0.1)
        ]),
    dbc.InputGroup([
        dbc.InputGroupAddon("Mortality rate", addon_type="prepend"),
        dbc.Input(id='id_ch_mort', type="number", value=0, min=-5, max=5, step=0.1)
        ]),
    dbc.InputGroup([
        dbc.InputGroupAddon("Life expectancy", addon_type="prepend"),
        dbc.Input(id='id_ch_life_exp', type="number", value=0, min=-5, max=5, step=0.05)
        ]),
    ])
))

'''
Creating the app layout and including the necessary content
'''

app.layout = dbc.Container([
    html.Div(
        children=[
            html.H1(children='Water scarcity'),
            html.H2(children='Predict the water scarcity in your country over the next years.')
            ], style = {'textAlign':'center', 'color':'SlateGrey'}
        ),
    html.Hr(),
    html.H5("Simulate future water scarcity levels for each country, with different climate and socio economic scenarios.",
        style = {'textAlign':'center', 'color':'SlateGrey'}),
    html.Div("All scenario percentage changes should be provided on a yearly basis.",
        style = {'textAlign':'left', 'color':'SlateGrey'}),
    html.Ul(children=[
        html.Li("Climate scenario: Level of increase in atmospheric CO2 levels."),
        html.Li("Population: Rate of population growth."),
        html.Li("Urbanisation: percentage change in the share of urban population."),
        html.Li("GDP per capita: percentage change in the GDP per capita."),
        html.Li("Mortality rate: percentage change in the infant mortality rate."),
        html.Li("Life expectancy: percentage change in the life expectancy.")
        ], style = {'textAlign':'left', 'color':'SlateGrey'}),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Row(html.Div("Select settings:")),
            dbc.Row(ins_changes)
            ], style = {'textAlign':'center'}),
        dbc.Col(dcc.Graph(id = 'id_plt_main', figure = fig))
        ], align = "center"),
    html.Footer(
        children = [
            html.Hr(),
            html.Div("KU Leuven: Modern Data Analystics - Team Tuvalu project - May 2021.")
        ], style = {'textAlign':'right'})
], fluid = True, style = {'backgroundColor':'AliceBlue'})


'''
App interactivity 
'''
@app.callback(
    Output('id_plt_main', 'figure'),
    Input('id_target_var', 'value'),
    Input('id_sel_country', 'value'),
    Input('id_sel_climate', 'value'),
    Input('id_ch_pop', 'value'),
    Input('id_ch_urban', 'value'),
    Input('id_ch_gdp', 'value'),
    Input('id_ch_mort', 'value'),
    Input('id_ch_life_exp', 'value')
)

def update_plot(target_var, country, climate, population, urban, gdp, mort, life_exp):
    
    preds = modelbackend.get_pred(target_var, country, climate, population, urban, gdp, mort, life_exp)
    
    fig = modelbackend.make_plot(preds)
    
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)