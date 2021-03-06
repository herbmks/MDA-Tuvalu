# This script includes useful tools that do not need to be explicitly shown in the main notebook/app.
# Some of the tools have been turned into classes as they perform similar tasks.
# This allows for better organisation and ease of use.


import pandas as pd
import numpy as np
import country_converter as coco


class data_cleaner():
    # Class that includes different methods which import and perform data cleaning on specific datasets.

    def socioecon_factors(self, aquastat_file_name, unicef_file_name):
        # Imports and cleans the socioeconimic factor datasets - aquastat and unicef
        
        # Aquastat
        df_aqua = pd.read_csv(aquastat_file_name)
        df_aqua = df_aqua.rename(columns = {'Unnamed: 0':'Country', 'Unnamed: 1':'Variable'})
        aqua_named_cols = [c for c in df_aqua.columns if c.lower()[:7] != 'unnamed']
        df_aqua = df_aqua[aqua_named_cols]
        df_aqua = df_aqua.dropna(subset = ['Country'])
        df_aqua = df_aqua.drop(['2018-2022'], axis = 1)
        aqua_missing_per_var = df_aqua.groupby(['Variable']).apply(lambda x: x.isnull().sum()).sum(axis = 1).sort_values()
        df_aqua = df_aqua[~df_aqua.isin(["Total water withdrawal per capita (m<sup>3</sup>/year per inhabitant)", 'Gender Inequality Index (GII) [equality = 0; inequality = 1) (-)']).any(axis=1)]
        df_aqua = df_aqua.dropna(subset = ['Variable'], axis = 0)
         
        # Unicef
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
        
        # Pivot tables + subset for 2013-2017
        df_unicef = df_unicef.pivot(index=['Country','Time'],columns='Indicator',values='Value')
        df_unicef.reset_index(inplace=True)
        df_unicef = df_unicef[(df_unicef['Time'] > 2012) & (df_unicef['Time'] < 2018)].groupby('Country').mean()
        df_unicef.drop(columns='Time',inplace=True)
        df_unicef.reset_index(inplace=True)
        df_aqua=df_aqua.drop(columns=['1998-2002','2003-2007','2008-2012'])
        df_aqua = df_aqua.pivot(index='Country',columns='Variable',values='2013-2017')
        df_aqua.reset_index(inplace=True)
        
        # Merge aqua and unicef
        df_aqua.set_index('Country',inplace=True)
        df_unicef.set_index('Country',inplace=True)
        df_socioec_factors = pd.merge(df_aqua,df_unicef,left_index=True,right_index=True)
        
        # Rename variables
        df_socioec_factors.rename(columns = {'Human Development Index (HDI) [highest = 1] (-)': 'HDI', 'Rural population (1000 inhab)' : 'rural_pop', 'Urban population (1000 inhab)' : 'urban_pop', 'Rural population with access to safe drinking-water (JMP) (%)' : 'rural_water', 'Urban population with access to safe drinking-water (JMP) (%)' : 'urban_water', 'Life expectancy at birth, total (years)':'life_ex', 'Mortality rate, infant (per 1,000 live births)': 'mort_rate', 'Population growth (annual %)': 'pop_growth', 'GDP per capita, PPP (constant 2011 international $)': 'GDP_pcp'}, inplace=True)
        
        # Add two additional variables
        df_socioec_factors['r_u'] = pd.to_numeric(df_socioec_factors['rural_pop']) / pd.to_numeric(df_socioec_factors['urban_pop'])
        df_socioec_factors['r_u_access'] = pd.to_numeric(df_socioec_factors['rural_water']) / pd.to_numeric(df_socioec_factors['urban_water'])
        
        df_socioec_factors.reset_index(inplace=True)
                
        return df_socioec_factors

    def climate_factors(self, rainfall_file_name, temperature_file_name, water_resources_file_name):
        # Imports and cleans the climate factor dataset
        
        # Rainfall - 2013-2017
        df_rain = pd.read_csv(rainfall_file_name, skipinitialspace = True).rename(columns = {'Rainfall - (MM)':'Total Rainfall (mm)'})
        df_rain.replace(to_replace = 'Congo (Republic of the)', value = 'Congo', inplace = True)
        df_rain['Country'] = coco.convert(names = df_rain['Country'], to = 'ISO3')
        df_rain = df_rain.rename(columns={'Statistics':'Month'})
        df_rain['Month'] = df_rain['Month'].map(lambda x: x.split(' ')[0])
        df_rain = df_rain[['Country','Year','Month','Total Rainfall (mm)']]
        df_rain = df_rain.groupby(['Country','Year']).sum()
        df_rain.reset_index(inplace=True)
        df_rain = df_rain[(df_rain['Year'] > 2012) & (df_rain['Year'] < 2018)].groupby('Country')['Total Rainfall (mm)'].mean()
        
        # Temperature - 2013-2017
        df_temp=pd.read_csv(temperature_file_name,skipinitialspace=True).rename(columns={'Temperature - (Celsius)':'Temperature (??C)'})
        df_temp.replace(to_replace = 'Congo (Republic of the)', value = 'Congo', inplace = True)
        df_temp['Country'] = coco.convert(names=df_temp['Country'], to='ISO3')
        df_temp = df_temp.rename(columns={'Statistics':'Month'})
        df_temp['Month'] = df_temp['Month'].map(lambda x: x.split(' ')[0])
        df_temp = df_temp[['Country','Year','Month','Temperature (??C)']]
        weights = {'Jan':31, 'Feb':28, 'Mar':31, 'Apr':30, 'May':31, 'Jun':30,'Jul':31, 'Aug':31, 'Sep':30, 'Oct':31, 'Nov':30, 'Dec':31}
        df_temp['Weight'] = df_temp['Month'].map(weights)
        df_temp = df_temp.groupby(['Country','Year']).apply(lambda x: (x['Temperature (??C)'] * x['Weight']).sum() / x['Weight'].sum())
        df_temp = pd.DataFrame(df_temp).rename(columns={0:'Temperature (??C)'})
        df_temp.reset_index(inplace=True)
        df_temp = df_temp[(df_temp['Year'] > 2012) & (df_temp['Year'] < 2018)].groupby('Country')['Temperature (??C)'].mean()

        # Water Resources - 2013-2017
        df_water = pd.read_csv(water_resources_file_name, nrows = 835, index_col = False).rename(columns = {'Area':'Country'})
        df_water.replace(to_replace = 'Grenade', value = 'Grenada', inplace = True)
        df_water['Country'] = coco.convert(names = df_water['Country'], to = 'ISO3')
        df_water = df_water.pivot(index = 'Country', columns = 'Variable Name', values = 'Value').rename(columns = {'Water resources: total external renewable':'Total external renewable water resources (ERWR)'})
        df_water = df_water[['Total internal renewable water resources (IRWR)','Total external renewable water resources (ERWR)','Total renewable water resources','Dependency ratio','Total exploitable water resources']]
        
        # Merge Factors
        df_climate_factors = pd.merge(df_temp,df_rain,left_index=True,right_index=True)
        df_climate_factors = pd.merge(df_climate_factors,df_water,left_index=True,right_index=True,how='outer')
        
        # Rename variables
        df_climate_factors.rename(columns={'Temperature (??C)':'Temp','Total Rainfall (mm)':'Rain','Total internal renewable water resources (IRWR)':'IRWR','Total external renewable water resources (ERWR)':'ERWR','Total renewable water resources':'TRWR','Dependency ratio':'Dep_ratio'}, inplace=True)
        
        # Recalculate water resource variables
        df_climate_factors['IRWR'].fillna(0,inplace=True)
        df_climate_factors['ERWR'].fillna(0,inplace=True)
        df_climate_factors[df_climate_factors['ERWR']<0] = 0
        df_climate_factors['TRWR'] = df_climate_factors['IRWR']+df_climate_factors['ERWR']
        df_climate_factors['Dep_ratio'] = df_climate_factors['ERWR']/df_climate_factors['TRWR']
        
        df_climate_factors.reset_index(inplace=True)
        
        return df_climate_factors
        
    def water_stress(self, water_stress_filename):
        # Imports and cleans the water stress indicator dataset
        
        # Water stress indicators - 2017
        df_waterstress = pd.read_csv(water_stress_filename,nrows=1970,index_col=False).rename(columns={'Area':'Country'})
        df_waterstress.replace(to_replace = 'Grenade', value = 'Grenada', inplace = True)
        df_waterstress['Country'] = coco.convert(names = df_waterstress['Country'], to='ISO3')
        df_waterstress = df_waterstress[df_waterstress['Year']==2017]
        df_waterstress = df_waterstress.pivot(index='Country', columns='Variable Name', values='Value')
        
        # Rename variables
        df_waterstress.rename(columns={'MDG 7.5. Freshwater withdrawal as % of total renewable water resources':'WS_MDG','SDG 6.4.2. Water Stress':'WS_SDG','SDG 6.4.1. Water Use Efficiency':'WUE_SDG' },inplace=True)    
             
        return df_waterstress

    
## This can be removed?
    def temp_proj(self, file_name):
        # Imports and cleans the forecasted temperatures

        df = pd.read_csv(file_name, sep = ',', header = 0, skipinitialspace = True, usecols = range(0,6))
        df.rename({df.columns[0]: "Temperature (??C)", df.columns[1]: "Year", df.columns[2]: "Model", df.columns[3]: "Month", df.columns[4]: "Name", df.columns[5]: "Country"}, axis = 1, inplace = True)
        df = df[['Country','Month','Temperature (??C)']]
        df['Month'] = df['Month'].map(lambda x: x.split(' ')[0])
        df = df[['Country','Month','Temperature (??C)']]
        weights = {'Jan':31, 'Feb':28, 'Mar':31, 'Apr':30, 'May':31, 'Jun':30,'Jul':31, 'Aug':31, 'Sep':30, 'Oct':31, 'Nov':30, 'Dec':31}
        df['Weight'] = df['Month'].map(weights)
        df = df.groupby('Country').apply(lambda x: (x['Temperature (??C)'] * x['Weight']).sum() / x['Weight'].sum())
        df = pd.DataFrame(df).rename(columns={0:'Temperature (??C)'})
        
        df.reset_index(inplace=True)
        
        return df