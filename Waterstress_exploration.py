#Libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from sklearn.decomposition import FactorAnalysis


# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))
os.chdir('D:/program files (x86)/Unief/2020-2021/Modern Data Analytics/Project')
print("Current working directory: {0}".format(os.getcwd()))


#Entering Data
data = pd.read_csv('waterstress_cleaned.csv', index_col= False) 
Waterstress_data = data.drop('Unnamed: 0', axis = 1)



#Exploration

Waterstress_data.columns = ['Name', 'Baseline_waterstress_total',
       'Baseline_waterstress_Agricultural', 'Baseline_waterstress_Domestic',
       'Baseline_waterstress_Industrial', 'Drought_severity_total',
       'Drought_severity_Agricultural', 'Drought_severity_Domestic',
       'Drought_severity_Industrial', 'Flood_occurence_total',
       'Flood_occurence_Agricultural', 'Flood_occurence_Domestic',
       'Flood_occurence_Industrial', 'Interannual_variability_total',
       'Interannual_variability_Agricultural',
       'Interannual_variability_Domestic',
       'Interannual_variability_Industrial', 'Seasonal_variability_total',
       'Seasonal_variability_Agricultural', 'Seasonal_variability_Domestic',
       'Seasonal_variability_Industrial', 'water_acces_total']

plt.hist(Waterstress_data['Baseline_waterstress_total'], bins = 25)
plt.hist(Waterstress_data['Baseline_waterstress_Agricultural'], bins = 25)
plt.hist(Waterstress_data['Baseline_waterstress_Domestic'], bins = 25)
plt.hist(Waterstress_data['Baseline_waterstress_Industrial'], bins = 25)

plt.hist(Waterstress_data['Drought_severity_total'], bins = 25)
plt.hist(Waterstress_data['Drought_severity_Agricultural'], bins = 25)
plt.hist(Waterstress_data['Drought_severity_Domestic'], bins = 25)
plt.hist(Waterstress_data['Drought_severity_Industrial'], bins = 25)

plt.hist(Waterstress_data['Flood_occurence_total'], bins = 25)
plt.hist(Waterstress_data['Flood_occurence_Agricultural'], bins = 25)
plt.hist(Waterstress_data['Flood_occurence_Domestic'], bins = 25)
plt.hist(Waterstress_data['Flood_occurence_Industrial'], bins = 25)

plt.hist(Waterstress_data['Interannual_variability_total'], bins = 25)
plt.hist(Waterstress_data['Interannual_variability_Agricultural'], bins = 25)
plt.hist(Waterstress_data['Interannual_variability_Domestic'], bins = 25)
plt.hist(Waterstress_data['Interannual_variability_Industrial'], bins = 25)

plt.hist(Waterstress_data['Seasonal_variability_total'], bins = 25)
plt.hist(Waterstress_data['Seasonal_variability_Agricultural'], bins = 25)
plt.hist(Waterstress_data['Seasonal_variability_Domestic'], bins = 25)
plt.hist(Waterstress_data['Seasonal_variability_Industrial'], bins = 25)

plt.hist(Waterstress_data['water_acces_total'], bins = 25)

#DataWithout Names Column
Waterstress_headless = Waterstress_data.drop('Name', axis = 1)



#Standardize
scaler = StandardScaler()
scaled_features = scaler.fit_transform(Waterstress_headless)
Waterstress_standardized = pd.DataFrame(scaled_features)
Waterstress_standardized.columns = Waterstress_headless.columns



#Creating Subdata

Waterstress_standardized_totals = Waterstress_standardized.iloc[:, lambda df:df.columns.str.contains('total', case=True)]
Waterstress_standardized_nototals = Waterstress_standardized.drop(Waterstress_standardized_totals.columns, axis = 1)
Waterstress_standardized_nototals['Water_acces'] = Waterstress_standardized_totals['water_acces_total']
#Missing Handling = Drop
Waterstress_standardized_nona = Waterstress_standardized.dropna()
Waterstress_standardized_totals_nona = Waterstress_standardized_totals.dropna()
Waterstress_standardized_nototals_nona = Waterstress_standardized_nototals.dropna()

        #Creating Correlation overview
corrMatrix = Waterstress_standardized_totals.corr()
corrMatrix = Waterstress_standardized.corr()        
corrMatrix = Waterstress_standardized_nototals_nona.corr()       
        
        #Creating a factor analysis (!Correlation between similar variables, try with Assorted (e.g. all agricultural))
Exploratory_Factor = FactorAnalysis(n_components=4, random_state=0)        
Factor_Results = Exploratory_Factor.fit_transform(Waterstress_standardized_nototals_nona)        
Factor_Loadings = pd.DataFrame(Exploratory_Factor.components_)
Factor_Loadings.columns = Waterstress_standardized_nototals_nona.columns



Exploratory_Factor = FactorAnalysis(n_components=2, random_state=0)        
Factor_Results = Exploratory_Factor.fit_transform(Waterstress_standardized_totals_nona)        
Factor_Loadings = pd.DataFrame(Exploratory_Factor.components_)
Factor_Loadings.columns = Waterstress_standardized_totals.columns

        #Creating a cluster analysis for countries
        

                ### FInding ClusterAmount
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(Waterstress_standardized_nototals_nona)
    sse.append(kmeans.inertia_)




plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)

        
 
    
 
                # 3 Clusters (or 8) seem most appropriate
                
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=50,
    max_iter=300,
    random_state=42
)    
Cluster_solution = kmeans.fit(Waterstress_standardized_nototals_nona)            
plt.hist(Cluster_solution.labels_)


res=Cluster_solution.__dict__
Centers = pd.DataFrame(res['cluster_centers_'])
Centers.columns = Waterstress_standardized_nototals_nona.columns

    
Labels = Cluster_solution.labels_
Waterstress_standardized_nototals_nona['Cluster_label'] = Labels
Waterstress_overview = Waterstress_standardized_nototals_nona.merge(Waterstress_data['Name'], left_index = True, right_index = True)
