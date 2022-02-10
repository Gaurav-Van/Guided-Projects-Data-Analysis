import folium
from folium import plugins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pywaffle import Waffle
from wordcloud import WordCloud, STOPWORDS

# -----------------------------------------
# MAPS AND GEOSPATIAL DATA
# -----------------------------------------

# 1) Folium - to analyze geospatial data , by default map style is open street map
# - folium.map()
# diff map styles - a) Stamen Toner - for rivers and coastal zones
# b) Stamen Terrain - for hills and natural vegetation colors
# c) Mapbox Bright - name of every country

delhi_map = folium.Map(location=[28.644800, 77.216721], zoom_start=5, tiles='Stamen Terrain')

# Markers
# circle mark at NH24

# feature groups for markers
# children - added to feature groups
NH24 = folium.map.FeatureGroup()
NH24.add_child(folium.features.CircleMarker((28.5946, 77.3241), radius=5, color='blue'))
delhi_map.add_child(NH24)

# label the marker
folium.Marker([28.5946, 77.3241], popup='NH 24').add_to(delhi_map)

delhi_map.save('Delhi.html')

# choropleth maps with folium - in which areas are shaded, according to data
# requires a Geo JSON file that includes geospatial data of the region

df_crime = pd.read_csv("C:\\Users\\Asus\\Downloads\\Cog =2\\Police_Department_Incidents_2016.csv")  # San Francisco
print(f'{df_crime.shape} \n {df_crime.columns} \n {df_crime["Category"]}')
df_crime_lim = df_crime.iloc[0:100, :]
print(df_crime_lim.shape)
# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

SanFran_map = folium.Map([latitude, longitude], zoom_start=13)
incidents = folium.map.FeatureGroup()
latit = list(df_crime_lim['Y'])
longi = df_crime_lim['X'].tolist()
labels = df_crime_lim['Category'].tolist()
print(f'\n {latit} \n {longi} \n {labels}')

for lat, long in zip(latit, longi):
    incidents.add_child(folium.features.CircleMarker((lat, long), radius=4, color='red'))

for lat, long, label in zip(latit, longi, labels):
    folium.Marker([lat, long], popup=label).add_to(SanFran_map)

SanFran_map.add_child(incidents)

SanFran_map.save('SanFran.html')

# Marker clusters

SanFran_map2 = folium.Map([latitude, longitude], zoom_start=10)
inc = plugins.MarkerCluster().add_to(SanFran_map2)
for lat, long, label in zip(latit, longi, labels):
    folium.Marker([lat, long], popup=label).add_to(SanFran_map2)
SanFran_map2.save('SanFran2.html')

# Ch Map
# A Choropleth map is a thematic map in which areas are shaded or patterned in proportion to the measurement of the statistical
# variable being displayed on the map, such as population density or per-capita income. The choropleth map provides an easy way to
# visualize how a measurement varies across a geographic area, or it shows the level of variability within a region.
# In our case, since we are endeavoring to create a world map, we want a GeoJSON that defines the boundaries of all world countries.
# we need GeoJSON file- contains information [geographic information] about countries and all

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json'
df = pd.read_json(url)
print(df)
df.to_json('world_countries.json')
world_geo_data = f'{url}'
# And now to create a Choropleth map, we will use the choropleth method with the following main parameters:
#
# geo_data, which is the GeoJSON file.
# data, which is the dataframe containing the data.
# columns, which represents the columns in the dataframe that will be used to create the Choropleth map.
# key_on, which is the key or variable in the GeoJSON file that contains the name of the variable of interest. To determine that,
# you will need to open the GeoJSON file using any text editor and note the name of the key or variable that contains the name of the countries,
# since the countries are our variable of interest. In this case, name is the key in the GeoJSON file that contains the name of the countries.
# Note that this key is case_sensitive, so you need to pass exactly as it exists in the GeoJSON file.

df_c_imm = pd.read_excel("C:\\Users\\Asus\\Downloads\\Cog =2\\Canada.xlsx", sheet_name='Canada by Citizenship',
                         skiprows=range(20),
                         skipfooter=2)

df_c_imm.columns = list(map(str, df_c_imm.columns))

df_c_imm.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
print(df_c_imm.head())

df_c_imm.rename(
    columns={'OdName': 'Country', 'AreaName': 'Continent', 'RegName': 'Region', 'DevName': 'Type of Region'},
    inplace=True)
df_c_imm['Total'] = df_c_imm.loc[:, '1980':'2013'].sum(axis=1)

world_map = folium.Map([0, 0], zoom_start=4)

folium.Choropleth(
    geo_data=world_geo_data,
    data=df_c_imm,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='RdYlGn',
    legend_name='Immigration to Canada'
).add_to(world_map)

world_map.save('World_Map.html')
