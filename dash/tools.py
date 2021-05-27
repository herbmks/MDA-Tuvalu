'''

This script includes the backend functionality of the Dash app.

'''
import pandas as pd
import numpy as np
import country_converter as coco

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from plotly.subplots import make_subplots
import plotly.graph_objects as go

class PredModels():
    """
    This class includes all the functionality of the main plot for the app.
    """


    def __init__(self):
        """Creates the self objects for the data and the predictive models."""
        self.df_full, self.df_pred, self.df_target = self.import_main_data()
        
        self.df_temp_pred, self.df_rain_pred = self.import_climate_pred_data()

        self.model_ws_mdg, self.model_wue_sdg, self.model_ws_sdg = self.make_models()


    def import_main_data(self):
        """Imports the dataset and creates the necessary dataframes"""
        df = pd.read_csv('final_data.csv', index_col = 0)

        df['IRWR_capita'] = df['IRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['ERWR_capita'] = df['ERWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['TRWR_capita'] = df['TRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)

        df_pred = df.iloc[:, 4:22]
        df_target = df.iloc[:, 1:4]

        return df, df_pred, df_target

    def import_climate_pred_data(self):
        """Imports the datasets with climate prediction values for temperature and rain"""
        
        df_temp = pd.read_csv('temperature_predictions.csv', index_col = 0)
        df_rain = pd.read_csv('rainfall_predictions.csv', index_col = 0)
        
        
        return df_temp, df_rain

    def make_models(self):
        """Creates and trains the predictive model pipes for the target variables."""
        
        # Creating pipe
        logscaler = FunctionTransformer(log_transform)
        scaler = ColumnTransformer(remainder = 'passthrough',
            transformers = [
                ("logscaler", logscaler, ['Rain', 'IRWR', 'ERWR', 'TRWR', 'IRWR_capita', 'ERWR_capita', 'TRWR_capita','rural_pop', 'urban_pop', 'GDP_pcp'])
            ])
        pca_pred = PCA()
        model = Ridge()
        model_pipe = Pipeline([
            ('scaler', scaler),
            ('reduce_dim', pca_pred),
            ('regressor', model)
        ])

        # Creating prediction models
        model_ws_mdg = model_pipe
        model_ws_mdg.set_params(**{'reduce_dim__n_components': 12,'regressor__alpha': 3.8000000000000003})
        model_ws_mdg.fit(self.df_pred, self.df_target['WS_MDG'])
        model_wue_sdg = model_pipe
        model_wue_sdg.set_params(**{'reduce_dim__n_components': 14, 'regressor__alpha': 2.7})
        model_wue_sdg.fit(self.df_pred, self.df_target['WUE_SDG'])
        model_ws_sdg = model_pipe
        model_ws_sdg.set_params(**{'reduce_dim__n_components': 12, 'regressor__alpha': 2.8000000000000003})
        model_ws_sdg.fit(self.df_pred, self.df_target['WS_SDG'])

        return model_ws_mdg, model_wue_sdg, model_ws_sdg

    def get_pred(self, target, country, climate, ch_pop, ch_urban, ch_gdp, ch_mort, ch_life_exp, years = 13):
        """Creates future predicitons for the scenario provided using the inputs."""
        
        # Create input matrix with future values
        # changes = [temp, rain, IRWR, ERWR, TRWR, dep_ratio, rural_pop, urban_pop, HDI, r_u, r_u_access, pop_growth, mort_rate, GDP_pcp, life_ex, IRWR_capita, ERWR_capita, TRWR_capita]
        current = np.asarray(self.df_full.iloc[(self.df_full['Country'] == country).values, 4:22])[0]
        temp_pred = np.asarray(self.df_temp_pred.iloc[((self.df_temp_pred['Country'] == country) & (self.df_temp_pred['type'] == climate)).values, 1:11])[0]
        rain_pred = np.asarray(self.df_rain_pred.iloc[((self.df_rain_pred['Country'] == country) & (self.df_rain_pred['type'] == climate)).values, 1:11])[0]
        
        current_pop = current[6] + current[7]
        current_urban_pc = current[7] / current_pop

        changes = np.asarray([0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1 + 0.01*ch_mort, 1 + 0.01*ch_gdp, 1 + 0.01*ch_life_exp, 0, 0, 0])

        mx_changes = np.zeros((years, len(changes)))
        pop = np.zeros((years,))
        urban_pc = np.zeros((years,))

        for i in range(years):
            mx_changes[i] = changes**(i)
            pop[i] = current_pop * ((1+0.01*ch_pop)**(i))
            urban_pc[i] = current_urban_pc + (ch_urban*(i))

        x_scenario = current * mx_changes
        
        x_scenario[0:3, 0] = np.repeat(current[0], 3)
        x_scenario[0:3, 1] = np.repeat(current[1], 3)
        
        x_scenario[3:, 0] = temp_pred
        x_scenario[3:, 1] = rain_pred
        
        urban_pc = np.clip(urban_pc, 0.0001, 0.9999)
        x_scenario[:, 6] = pop * (1 - urban_pc)
        x_scenario[:, 7] = pop * urban_pc

        x_scenario[:, 9] = (pop * (1 - urban_pc))/(pop * urban_pc)

        x_scenario[:, 11] = np.repeat(ch_pop, years)
        
        x_scenario[:, 12] = np.clip(x_scenario[:, 12], 0, 1000)
        
        x_scenario[:, 15] = current[2] / (pop * 1000)
        x_scenario[:, 16] = current[3] / (pop * 1000)
        x_scenario[:, 17] = current[4] / (pop * 1000)
        
        x_scenario = np.around(x_scenario, decimals = 7)
        
        # Turn into pandas df
        cols = list(self.df_pred.columns)
        x_scenario = pd.DataFrame(x_scenario, columns = cols)
        
        
        # Generate prediction
        if target == 'WS_MDG':
            y_pred = self.model_ws_mdg.predict(x_scenario)

        elif target == 'WUE_SDG':
            y_pred = self.model_wue_sdg.predict(x_scenario)

        elif target == 'WS_SDG':
            y_pred = self.model_ws_sdg.predict(x_scenario)

        return y_pred

    def make_plot(self, pred_data, indicator):
        """Creates a plot of the future predicitons."""

        fig = make_subplots(rows = 1, cols = 1)

        fig.add_trace(
            go.Scatter(x = np.arange(2017, 2030, 1),
            y = pred_data,
            name = 'Prediction'),
        row=1, col=1)

        fig.update_layout(
            width = 1000,
            title = "Prediction",
            xaxis_title = "Year",
            yaxis_title = ("Water Scarcity Indicator (" + indicator + ")"),
            xaxis = dict(
                tickmode = 'linear',
                tick0 = 2017,
                dtick = 1
            )
        )

        return fig

    def get_country_dict(self):
        """Generates dictionary used in country choice input field."""
        codes = self.df_full.iloc[:, 0]
        codes = list(codes)

        countries = coco.convert(names = codes, to = 'name_short')

        dict_list = []
        for i in range(len(codes)):
            iter = {"label": countries[i], 'value': codes[i]}
            dict_list.append(iter)

        return dict_list



def log_transform(x):
    return np.log(x+1)