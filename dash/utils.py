'''

Backend functions for the app.

'''
import pandas as pd
import numpy as np
import country_converter as coco
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decompostion import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


class PredModel():
    
    def __init__(self):
        self.df_pred, self.df_target = self.import_data()
        
        self.model_pipe = self.make_model_pip()
        
        self.fit_model()
        
    def import_data(self):
        '''Imports the dataset'''
        df = pd.read_csv('final_data.csv', index_col = 0)
        
        df['IRWR_capita'] = df['IRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['ERWR_capita'] = df['ERWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        df['TRWR_capita'] = df['TRWR'] / ((df['urban_pop'] + df['rural_pop']) * 1000)
        
        df_pred = df.iloc[:, 4:22]
        df_target = df.iloc[:, 1:4]
        
        return df_pred, df_target

    def make_model_pipe(self):
        
        logscaler = FunctionTransformer(log_transform)
        
        scaler = ColumnTransformer(remainder = 'passthrough',
            transformers = [
                ("logscaler", logscaler, ['Rain', 'IRWR', 'ERWR', 'TRWR', 'IRWR_capita', 'ERWR_capita', 'TRWR_capita','rural_pop', 'urban_pop', 'GDP_pcp'])
            ])
        
        pca_pred = PCA()
        
        model = RandomForestRegressor(random_state = 0)
        
        model_pipe = Pipeline([
            ('scaler', scaler),
            ('reduce_dim', pca_pred),
            ('regressor', model)
        ])
        
        return model_pipe

    def fit_model(self):
        
        
        
        
        
        self.model_pipe.fit()
        
        
    
    def get_pred(self, params):
    
    

    
    
    
    def make_plot(self, )


    def get_country_dict(self):
        
        
        return dict 
        


def log_transform(x):
    return np.log(x+1)