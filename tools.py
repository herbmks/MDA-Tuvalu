"""
This script includes useful tools that do not need to be explicitly shown in the main notebook/app.
Some of the tools have been turnde into classes as they perform similar tasks.
This allows for better organisation and ease of use.
"""

import pandas as pd
import country_converter as coco


class data_cleaner():
    """ Class that includes different methods which import and perform data cleaning on specific datasets. """
    
    def temp_forecast(file_name):
        """Imports and cleans the forecasted temperatures"""
    
        df = pd.read_csv(file_name, sep = ',', header = 0, usecols = range(0,6))
        df.rename({df.columns[0]: "Temperature", df.columns[1]: "Year", df.columns[2]: "Model", df.columns[3]: "Month", df.columns[4]: "Country", df.columns[5]: "Code"}, axis = 1, inplace = True)

        df['Code'] = df['Code'].str.strip()
        df['Country'] = df['Country'].str.strip()
        df['Model'] = df['Model'].str.strip()
    
        name = df.Month.unique()
        number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        for i in range(0,12):
            df = df.replace(name[i], number[i])
    
        return df 
    
    def temp_hist(file_name, names_years = False):
        """Imports and cleans the historical temperatures"""
        df = pd.read_csv(file_name, sep = ',', header = 0, usecols = range(0,5))
        df.rename({df.columns[0]: "Temperature", df.columns[1]: "Year", df.columns[2]: "Month", df.columns[3]: "Country", df.columns[4]: "Code"}, axis = 1, inplace = True)

        df['Code'] = df['Code'].str.strip()
        df['Country'] = df['Country'].str.strip()
    
        name = df.Month.unique()
        number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        for i in range(0,12):
            df = df.replace(name[i], number[i])
    
        if names_years is False:    
            return df
    
        country_codes = np.asarray(pd.unique(df.Code))
        years = np.asarray(pd.unique(df.Year))
    
        return clean_data, country_codes, years
    
    def socioecon_factors(aquastat_file_name, unicef_file_name):
        """Imports and cleans the socio econimic factor datasets - aquastat and unicef """
        
        if aquastat_file_name is None:
            return print("Provide file name for AQUASTAT dataset")
        
        df_aqua = pd.read_csv('aquastat_file_name')
        
        df_aqua = df_aqua.rename(columns = {'Unnamed: 0':'Country', 'Unnamed 1':'Variable'})
        aqua_named_cols = [c for c in df_aqua.columns if c.lower()[:7] != 'unnamed']
        df_aqua = df_aqua[aqua_named_cols]
        
        df_aqua = df_aqua.dropna(subset = ['Country'])
        df_aqua = df_aqua.drop(['2018-2022'], axis = 1)        
        df_aqua = df_aqua[~DF_aqua.isin(["Total water withdrawal per capita (m<sup>3</sup>/year per inhabitant)", 'Gender Inequality Index (GII) [equality = 0; inequality = 1) (-)']).any(axis=1)]
        df_aqua = df_aqua.dropna(subset = ['Variable'], axis = 0)
        
        if unicef_file_name is None:
            return print("Provide file name for UNICEF dataset")
        
        df_unicef = pd.read_csv('unicef_file_name')
        
        unicef_keep_cols = ['Indicator', 'Country', 'Time', 'Value']
        df_unicef = df_unicef[unicef_keep_cols]
        
        unicef_missing_per_var = df_unicef.groupby(['Indicator']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        remove_vars = missing_per_var[ unicef_missing_per_var > 60].index
        df_unicef = df_unicef[ ~df_unicef.isin(remove_vars).any(axis = 1)]
        
        aqua_missing_per_country = df_aqua.groupby(['Country']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        aqua_countries = aqua_missing_per_country[aqua_missing_per_country == 0].index
        df_aqua = df_aqua[df_aqua['Country'].isin(aqua_countries)]
        
        unicef_missing_per_country = df_unicef.groupby(['Country']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        unicef_countries =  unicef_missing_per_country[unicef_missing_per_country == 0].index
        df_unicef = df_unicef[df_unicef['Country'].isin(unicef_countries)]
        
        # Replacing country names by ISO3 country codes.
        
        df_aqua['Country'] = coco.convert(names = df_aqua['Country'], to = 'ISO3', not_found = 'XXX')
        df_aqua = df_aqua[~df_aqua.isin(['XXX']).any(axis = 1)]
        
        df_unicef = coco.convert(names = df_unicef['Country'], to = 'IOS3', not_found = 'XXX')
        df_unicef = df_unicef[~df_unicef['Country'].isin(['XXX'])]
        
        # Removing countries that only exist in one of the two datasets
        
        df_aqua = df_aqua[df_aqua['Country'].isin(df_unicef['Country'].unique())]
        df_unicef = df_unicef[df_unicef['Country'].isin(df_aqua['Country'].unique())]
        
        return df_aqua, df_unicef
    
    def climate_factors(file_name):
        """Imports and cleans the climate factor dataset"""
        
        
        
        return df_climate
    
    
    