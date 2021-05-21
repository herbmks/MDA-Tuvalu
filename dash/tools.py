'''

Backend functions for the app.

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

    def __init__(self):
        self.df_full, self.df_pred, self.df_target = self.import_data()

        #self.model_ws_mdg, self.model_wue_sdg, self.model_ws_sdg = self.make_models()


    def import_data(self):
        '''Imports the dataset'''
        df = pd.read_csv('final_data.csv', index_col = 0)

        df['IRWR_capita'] = df['IRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['ERWR_capita'] = df['ERWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['TRWR_capita'] = df['TRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)

        df_pred = df.iloc[:, 4:22]
        df_target = df.iloc[:, 1:4]

        return df, df_pred, df_target

    def make_models(self):

        logscaler = FunctionTransformer(log_transform)

        scaler = ColumnTransformer(remainder = 'passthrough',
            transformers = [
                ("logscaler", logscaler, ['Rain', 'IRWR', 'ERWR', 'TRWR', 'IRWR_capita', 'ERWR_capita', 'TRWR_capita','rural_pop', 'urban_pop', 'GDP_pcp'])
            ])

        pca_pred = PCA()
        n_comp_test = np.arange(3, 15)

        model = Ridge()
        alphas_test = np.power(10, np.arange(-2, 0, 0.02))

        model_pipe = Pipeline([
            ('scaler', scaler),
            ('reduce_dim', pca_pred),
            ('regressor', model)
        ])
        
        test_params = [{
            'reduce__n_components': n_comp_test.
            'regressor__alpha': alphas_test
            }]
        
        gridsearch_ws_mdg = GridSearchCV(pipe_climate, test_params, verbose=1, n_jobs=-1).fit(np.asarray(self.df_pred), np.asarray(self.df_target['WS_MDG']))
        gridsearch_wue_sdg = GridSearchCV(pipe_climate, test_params, verbose=1, n_jobs=-1).fit(np.asarray(self.df_pred), np.asarray(self.df_target['WUE_SDG']))
        gridsearch_ws_sdg = GridSearchCV(pipe_climate, test_params, verbose=1, n_jobs=-1).fit(np.asarray(self.df_pred), np.asarray(self.df_target['WS_SDG']))
        
        model_ws_mdg = gridsearch_ws_mdg.best_estimator_
        model_ws_mdg.fit(np.asarray(self.df_pred), np.asarray(self.df_target['WS_MDG']))
        model_wue_sdg = gridsearch_wue_sdg.best_estimator_
        model_wue_sdg.fit(np.asarray(self.df_pred), np.asarray(self.df_target['WUE_SDG']))
        model_ws_sdg = gridsearch_ws_sdg.best_estimator_
        model_ws_sddg.fit(np.asarray(self.df_pred), np.asarray(self.df_target['WS_SDG']))

        return model_ws_mdg, model_wue_sdg, model_ws_sdg

    def get_pred(self, params):
        
        
        
        
        return print("w.i.p.")

    def make_plot(self):

        return print("w.i.p.")

    def get_country_dict(self):

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