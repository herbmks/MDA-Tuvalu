"""
This script includes useful tools that do not need to be explicitly shown in the main notebook/app.
Some of the tools have been turnde into classes as they perform similar tasks.
This allows for better organisation and ease of use.
"""

import pandas as pd
import country_converter as coco


class data_cleaner():
    """ Class that includes different methods which import and perform data cleaning on specific datasets. """

    def temp_forecast(self, file_name):
        """Imports and cleans the forecasted temperatures"""

        df = pd.read_csv(file_name, sep = ',', header = 0, usecols = range(0,6))
        df.rename({df.columns[0]: "Temperature", df.columns[1]: "Year", df.columns[2]: "Model", df.columns[3]: "Month", df.columns[4]: "Country", df.columns[5]: "Code"}, axis = 1, inplace = True)

        df['Code'] = df['Code'].str.strip()
        df['Country'] = df['Country'].str.strip()
        df['Model'] = df['Model'].str.strip()

        name = df.Month.unique()
        number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        months = dict(zip(name, number))

        df = df.replace({'Month': months})

        return df

    def temp_hist(self, file_name, names_years = False):
        """Imports and cleans the historical temperatures"""
        df = pd.read_csv(file_name, sep = ',', header = 0, usecols = range(0,5))
        df.rename({df.columns[0]: "Temperature", df.columns[1]: "Year", df.columns[2]: "Month", df.columns[3]: "Country", df.columns[4]: "Code"}, axis = 1, inplace = True)

        df['Code'] = df['Code'].str.strip()
        df['Country'] = df['Country'].str.strip()

        name = df.Month.unique()
        number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        months = dict(zip(name, number))

        df = df.replace({'Month': months})

        if names_years is False:
            return df

        country_codes = np.asarray(pd.unique(df.Code))
        years = np.asarray(pd.unique(df.Year))

        return df, country_codes, years

    def socioecon_factors(self, aquastat_file_name, unicef_file_name):
        """Imports and cleans the socioeconimic factor datasets - aquastat and unicef """
        df_aqua = pd.read_csv(aquastat_file_name)

        df_aqua = df_aqua.rename(columns = {'Unnamed: 0':'Country', 'Unnamed: 1':'Variable'})
        aqua_named_cols = [c for c in df_aqua.columns if c.lower()[:7] != 'unnamed']
        df_aqua = df_aqua[aqua_named_cols]
        df_aqua = df_aqua.dropna(subset = ['Country'])
        df_aqua = df_aqua.drop(['2018-2022'], axis = 1)
        aqua_missing_per_var = df_aqua.groupby(['Variable']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        df_aqua = df_aqua[~df_aqua.isin(["Total water withdrawal per capita (m<sup>3</sup>/year per inhabitant)", 'Gender Inequality Index (GII) [equality = 0; inequality = 1) (-)']).any(axis=1)]
        df_aqua = df_aqua.dropna(subset = ['Variable'], axis = 0)

        df_unicef = pd.read_csv(unicef_file_name)
        unicef_keep_cols = ['Indicator', 'Country', 'Time', 'Value']
        df_unicef = df_unicef[unicef_keep_cols]
        unicef_missing_per_var = df_unicef.groupby(['Indicator']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        remove_vars = unicef_missing_per_var[ unicef_missing_per_var > 60].index
        df_unicef = df_unicef[ ~df_unicef.isin(remove_vars).any(axis = 1)]

        aqua_missing_per_country = df_aqua.groupby(['Country']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        aqua_countries = aqua_missing_per_country[aqua_missing_per_country == 0].index
        df_aqua = df_aqua[df_aqua['Country'].isin(aqua_countries)]
        unicef_missing_per_country = df_unicef.groupby(['Country']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        unicef_countries =  unicef_missing_per_country[unicef_missing_per_country == 0].index
        df_unicef = df_unicef[df_unicef['Country'].isin(unicef_countries)]

        # Replacing country names by ISO3 country codes.
        df_aqua.replace(to_replace = 'Grenade', value = 'Grenada', inplace = True)
        df_unicef.replace(to_replace = 'Grenade', value = 'Grenada', inplace = True)

        df_aqua = df_aqua.drop(df_aqua[df_aqua.Country == 'Channel Islands'].index)
        df_unicef = df_unicef.drop(df_unicef[df_unicef.Country == 'Channel Islands'].index)

        df_aqua['Country'] = coco.convert(names = df_aqua['Country'], to = 'ISO3', not_found = 'XXX')
        df_aqua = df_aqua[~df_aqua.isin(['XXX']).any(axis = 1)]

        df_unicef['Country'] = coco.convert(names = df_unicef['Country'], to = 'ISO3', not_found = 'XXX')
        df_unicef = df_unicef[~df_unicef.isin(['XXX']).any(axis = 1)]

        # Removing countries that only exist in one of the two datasets
        C_aqua = df_aqua['Country'].unique()
        C_unicef = df_unicef['Country'].unique()
        C_shared = set.intersection(set(C_aqua), set(C_unicef))

        df_aqua = df_aqua[df_aqua['Country'].isin(C_shared)]
        df_unicef = df_unicef[df_unicef['Country'].isin(C_shared)]

        df_aqua.reset_index()
        df_unicef.reset_index()

        return df_aqua, df_unicef

    def climate_factors(self, rainfall_file_name, water_inflow_file_name):
        """Imports and cleans the climate factor dataset"""

        df_rain = pd.read_csv(rainfall_file_name, skipinitialspace = True).rename(columns = {'Rainfall - (MM)':'Total Rainfall (mm)'})
        df_rain.replace(to_replace = 'Congo (Republic of the)', value = 'Congo', inplace = True)
        df_rain['Country'] = coco.convert(names = df_rain['Country'], to = 'ISO3')

        df_inflow = pd.read_csv(water_inflow_file_name, nrows = 835, index_col = False).rename(columns = {'Area':'Country'})
        df_inflow.replace(to_replace = 'Grenade', value = 'Grenada', inplace = True)
        df_inflow['Country'] = coco.convert(names = df_inflow['Country'], to = 'ISO3')
        df_inflow = df_inflow.pivot(index = 'Country', columns = 'Variable Name', values = 'Value').rename(columns = {'Water resources: total external renewable':'Total external renewable water resources (ERWR)'})
        df_inflow = df_inflow[['Total internal renewable water resources (IRWR)','Total external renewable water resources (ERWR)','Total renewable water resources','Dependency ratio','Total exploitable water resources']]

        return df_rain, df_inflow


def rainfall_time(df_rain, adjustment = None):
    possible_adjustment = ['Monthly', 'Yearly', 'Average']

    if adjustment not in possible_adjustment:
        raise ValueError('adjustment must be one of - Monthly, Yearly, Average')

    df_monthly = df_rain[['Country', 'Year', 'Statistics', 'Total Rainfall (mm)']]
    df_monthly = df_monthly.rename(columns = {'Statistics':'Month'})
    df_monthly['Month'] = df_monthly['Month'].map(lambda x: x.split(' ')[0])
    df_monthly.set_index(['Country','Year','Month'],inplace = True)

    if adjustment is 'Monthly':
         return df_monthly

    df_yearly = df_monthly.groupby(['Country', 'Year']).sum()

    if adjustment is 'Yearly':
        return df_yearly

    if adjustment is 'Average':
        df_average = df_yearly.groupby(['Country']).mean()
        df_average['Standard Deviation'] = df_yearly.groupby(['Country']).std()

        return df_average
