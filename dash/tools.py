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


class PredModels():
    """
    This class includes all the functionality of the main plot for the app.
    """
    
    
    def __init__(self):
        """Creates the self objects for the data and the predicitve models."""
        self.df_full, self.df_pred, self.df_target = self.import_data()

        self.model_ws_mdg, self.model_wue_sdg, self.model_ws_sdg = self.make_models()


    def import_data(self):
        """Imports the dataset and creates the necessary dataframes"""
        df = pd.read_csv('final_data.csv', index_col = 0)

        df['IRWR_capita'] = df['IRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['ERWR_capita'] = df['ERWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['TRWR_capita'] = df['TRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)

        df_pred = df.iloc[:, 4:22]
        df_target = df.iloc[:, 1:4]

        return df, df_pred, df_target

    def make_models(self):
        """Creates and trains the predicitve model pipes for the target variables."""
        logscaler = FunctionTransformer(log_transform)

        scaler = ColumnTransformer(remainder = 'passthrough',
            transformers = [
                ("logscaler", logscaler, ['Rain', 'IRWR', 'ERWR', 'TRWR', 'IRWR_capita', 'ERWR_capita', 'TRWR_capita','rural_pop', 'urban_pop', 'GDP_pcp'])
            ])

        pca_pred = PCA()
        #n_comp_test = np.arange(3, 15)

        model = Ridge()
        #alphas_test = np.arange(0.1, 20, 0.1)

        model_pipe = Pipeline([
            ('scaler', scaler),
            ('reduce_dim', pca_pred),
            ('regressor', model)
        ])
        
        '''
        test_params = [{
            'scaler': [scaler],
            'reduce_dim__n_components': n_comp_test,
            'regressor': [model],
            'regressor__alpha': alphas_test
            }]
        
        gridsearch_ws_mdg = GridSearchCV(model_pipe, test_params, verbose=1, n_jobs=-1).fit(self.df_pred, self.df_target['WS_MDG'])
        gridsearch_wue_sdg = GridSearchCV(model_pipe, test_params, verbose=1, n_jobs=-1).fit(self.df_pred, self.df_target['WUE_SDG'])
        gridsearch_ws_sdg = GridSearchCV(model_pipe, test_params, verbose=1, n_jobs=-1).fit(self.df_pred, self.df_target['WS_SDG'])
        
        model_ws_mdg = gridsearch_ws_mdg.best_estimator_
        model_ws_mdg.fit(self.df_pred, self.df_target['WS_MDG'])
        model_wue_sdg = gridsearch_wue_sdg.best_estimator_
        model_wue_sdg.fit(self.df_pred, self.df_target['WUE_SDG'])
        model_ws_sdg = gridsearch_ws_sdg.best_estimator_
        model_ws_sdg.fit(self.df_pred, self.df_target['WS_SDG'])
        '''
        
        model_ws_mdg = model_pipe
        model_ws_mdg.fit(self.df_pred, self.df_target['WS_MDG'], {'reduce_dim__n_components': 12,'regressor__alpha': 3.8000000000000003})
        model_wue_sdg = model_pipe
        model_wue_sdg.fit(self.df_pred, self.df_target['WUE_SDG'], {'reduce_dim__n_components': 14, 'regressor__alpha': 2.7})
        model_ws_sdg = model_pipe
        model_ws_sdg.fit(self.df_pred, self.df_target['WS_SDG'], {'reduce_dim__n_components': 12, 'regressor__alpha': 2.8000000000000003})
        
        return model_ws_mdg, model_wue_sdg, model_ws_sdg

    def get_pred(self, target, country, climate, ch_pop, ch_urban, ch_gdp, ch_mort, ch_life_exp):
        """Creates future predicitons for the scenario provided using the inputs."""
        
        # current values for country in question
        current = np.asarray(self.df_full.loc[self.df_full['Country'] == country, 4:22])
                
        # population
        current_pop = current[6] + current[7]
        current_urban_pc = current[7] / current_pop
        
        # changes = [temp, rain, IRWR, ERWR, TRWR, dep_ratio, rural_pop, urban_pop, HDI, r_u, r_u_access, pop_growth, mort_rate, GDP_pcp, life_ex, IRWR_capita, ERWR_capita, TRWR_capita]        
        changes = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1 + 0.01*ch_mort, 1 + 0.01*ch_gdp, 1 + 0.01*ch_life_exp, 0, 0, 0]        
        
        mx_change = np.zeros((10, len(changes)))
        pop = np.zeros((10,))
        urban_pc = np.zeros((10,))
        
        for i in range(10):
            mx_change[i] = changes**i
            pop[i] = current_pop * (1+0.01*ch_pop)**i
            urban_pc[i] = current_urban_pc * (1+0.01*ch_urban)**i
            
        mx_changes[:, 7] = pop * urban_pc
        mx_changes[:, 6] = pop * (1 - urban_pc)
        
        mx_changes[:, 15] = current[15] / (pop * 100)
        mx_changes[:, 16] = current[16] / (pop * 100)
        mx_changes[:, 17] = current[17] / (pop * 100)
        
        
        x_scenario = current * mx_changes
        
        cols = self.df_pred.columns
        x_scenario = pd.DataFrame(x_scenario, columns = cols)
        
        if target == 'WS_MDG':
            y_pred = self.model_ws_mdg.predict(x_scenario)
        
        elif target == 'WUE_SDG':
            y_pred = self.model_wue_sdg.predict(x_scenario)
            
        elif target == 'WS_SDG':
            y_pred = self.model_ws_sdg.predict(x_scenario)
        
        return y_pred

    def make_plot(self, pred_data):
        """Creates a plot of the future predicitons."""
        
        fig = make_subplots(rows = 1, cols = 1)

        fig.add_trace(
            go.Scatter(x = pred_data,
            y = np.arange(2020, 2030, 1),
            name = 'Prediction'),
        row=1, col=1)
        
        fig.update_layout(width = 800)
        
        return print("w.i.p.")

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