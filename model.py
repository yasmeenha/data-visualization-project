import pandas as pd 
from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request, jsonify
#importing plotly and matplotlib for visualization charts
import matplotlib.pyplot as plt

import plotly_express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#here giving the dimensions for the graph
plt.rcParams["figure.figsize"] = (10,10)
plt.style.use('ggplot')
#reading the obesity ,world and populationdatasets
obesity_data = pd.read_csv('obesity_country.csv',   index_col=0)
population_data = pd.read_csv('world_population.csv')
world_region = pd.read_csv('world.csv')

global high, low, reg
#here creating the global map for visualization which shows all countries
map = {
    'Bahamas, The': 'Bahamas',
    'Bolivia': 'Bolivia (Plurinational State of)',
    'Congo, Rep.': 'Congo',
    'Czech Republic':'Czechia',
    "Cote d'Ivoire":"Côte d'Ivoire",
    'Korea, Dem. People’s Rep.':"Democratic People's Republic of Korea", 
    'Congo, Dem. Rep.':'Democratic Republic of the Congo',
    'Egypt, Arab Rep.':'Egypt',
    'Gambia, The':'Gambia',
    'Iran, Islamic Rep.':'Iran (Islamic Republic of)',
    'Kyrgyz Republic':'Kyrgyzstan',
    'Lao PDR':"Lao People's Democratic Republic",
    'Micronesia, Fed. Sts.':'Micronesia (Federated States of)',
    'Korea, Rep.':'Republic of Korea',
    'Moldova':'Republic of Moldova',
    'North Macedonia':'Republic of North Macedonia',
    'St. Kitts and Nevis':'Saint Kitts and Nevis',
    'St. Lucia':'Saint Lucia',
    'St. Vincent and the Grenadines':'Saint Vincent and the Grenadines',
    'Slovak Republic':'Slovakia',
    'Sudan':'Sudan (former)',
    'United Kingdom':'United Kingdom of Great Britain and Northern Ireland',
    'Tanzania':'United Republic of Tanzania',
    'United States':'United States of America',
    'Venezuela, RB':'Venezuela (Bolivarian Republic of)',
    'Vietnam':'Viet Nam',
    'Yemen, Rep.':'Yemen'
}

#giving specifications for creating a map showing all countries
map_new = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Bosnia and Herzegovina': 'Bosnia And Herzegovina',
    'Czechia': 'Czech Republic',
    "Côte d'Ivoire": "Côte D'Ivoire",
    "Democratic People's Republic of Korea": 'Korea, Republic of',
    'Democratic Republic of the Congo': 'Congo (Democratic Republic Of The)',
    'Guinea-Bissau': 'Guinea Bissau',
    'Iran (Islamic Republic of)': 'Iran',
    "Lao People's Democratic Republic": 'Laos',
    'Republic of Korea': 'South Korea',
    'Republic of Moldova': 'Moldova',
    'Republic of North Macedonia': 'Macedonia',
    'Russian Federation': 'Russia',
    'Sudan (former)': 'Sudan',
    'Syrian Arab Republic': 'Syria',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'United Republic of Tanzania': 'Tanzania',
    'United States of America': 'United States',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Viet Nam': 'Vietnam'
}


#selecting the important features or columns for visualization
obesity_data.columns = ['country', 'year', 'obesity', 'sex']
obesity_data.drop(obesity_data[obesity_data.country.isin(['South Sudan', 'Sudan', 'San Marino', 'Monaco'])].index, inplace=True)
#selecting obesity features that have to be shown on gra[hs]
obesity_data['obesity_prev'] = obesity_data.obesity.apply(lambda x: float(x.split(' ')[0]))
obesity_data['obesity_cri_lower'] = obesity_data.obesity.apply(lambda x: float((x.split(' ')[1]).split('-')[0][1:]))
obesity_data['obesity_cri_upper'] = obesity_data.obesity.apply(lambda x: float((x.split(' ')[1]).split('-')[1][:-1]))
obesity_data['obesity_cri_width'] = obesity_data['obesity_cri_upper'] - obesity_data['obesity_cri_lower']
#then replacing the column in population with country name
population_data.replace({'Country Name': map}, inplace=True)
obesity_data = obesity_data.merge(population_data, how='left', left_on=['country', 'year'], right_on=['Country Name', 'Year']).drop(['Country Name', 'Year'], axis=1)
obesity_data.rename(columns={'Count': 'population'}, inplace=True)
#then reading the obesity count and population
obesity_data['obesity_prev_count'] = obesity_data['obesity_prev'] / 100 * obesity_data['population']
obesity_data['obesity_cri_min_count'] = obesity_data['obesity_cri_lower'] / 100 * obesity_data['population']
obesity_data['obesity_cri_max_count'] = obesity_data['obesity_cri_upper'] / 100 * obesity_data['population']
#then reading obesity of each country
obesity_data['country_2'] = obesity_data['country'].values.copy()

obesity_data.replace({'country_2': map_new}, inplace=True)
#reading the world population data and reading 3 coumns name,region,subregion
world_region = world_region[['name', 'region', 'sub-region']]
obesity_data = obesity_data.merge(world_region, how='left', left_on=['country_2'], right_on=['name']).drop('name', axis=1)
#reading the obesity in males
obesity_data_male = obesity_data.loc[obesity_data.sex == 'Male', :].reset_index(drop=True)
obesity_data_male_pivot = obesity_data_male[['country', 'year', 'obesity_prev']].pivot(index='country', columns='year', values='obesity_prev')
#reading obesity in females
obesity_data_female = obesity_data.loc[obesity_data.sex == 'Female', :].reset_index(drop=True)
obesity_data_female_pivot = obesity_data_female[['country', 'year', 'obesity_prev']].pivot(index='country', columns='year', values='obesity_prev')
#combining both male and female obesity
obesity_data_both = obesity_data.loc[obesity_data.sex == 'Both sexes', :].reset_index(drop=True)
obesity_data_both_pivot = obesity_data_both[['country', 'year', 'obesity_prev']].pivot(index='country', columns='year', values='obesity_prev')
#calculating highest obese countries
high_obesity_countries = obesity_data_both.groupby('country').mean().sort_values(by='obesity_prev', ascending=False)['obesity_prev'][:20]
countries_high = high_obesity_countries.index.tolist()
values_high = high_obesity_countries.values.tolist()

#calculating countries with low obesity
low_obesity_countries = obesity_data_both.groupby('country').mean().sort_values(by='obesity_prev', ascending=True)['obesity_prev'][:20].sort_values(ascending=False)
countries_low = low_obesity_countries.index.tolist()
values_low = low_obesity_countries.values.tolist()

#calculating obesity by region
obesity_region = obesity_data_both.groupby(['region', 'year']).mean().reset_index(drop=False)
obesity_sub = obesity_data_both.groupby(['sub-region', 'year']).mean().reset_index(drop=False)

obesity_region_pivot = obesity_region[['region', 'year', 'obesity_prev']].pivot(index='region', columns='year', values='obesity_prev')
obesity_sub_pivot = obesity_sub[['sub-region', 'year', 'obesity_prev']].pivot(index='sub-region', columns='year', values='obesity_prev')

obesity_region_mean = obesity_data.groupby('region').mean().reset_index(drop=False)[['region', 'obesity_prev']].sort_values(by='obesity_prev', ascending=False)
obesity_subregion_mean = obesity_data.groupby('sub-region').mean().reset_index(drop=False)[['sub-region', 'obesity_prev']].sort_values(by='obesity_prev', ascending=False)
#then creating the visualization charts for each feature
class Chart:
#giving the commands for high obesity values
    def highest(self):
        global high
        high = px.bar(countries_high, x=countries_high, y=values_high, color=countries_high,
             title='Countries With Highest Obesity in the World',
             labels={'x': '', 'y': 'Average Obesity (%)'})

        high.update_layout(xaxis={'tickangle': 45, 'tickfont': {'size': 10},
                         'tickmode': 'linear', 'title': ''})
        high.update_layout(width=450, height=450)
        high.update_layout(showlegend=False)

        return high
#calculating lowest obese countries   
    def lowest(self):
        global low
        low = px.bar(countries_low,x=countries_low, y=values_low, color=countries_low,
             title='Countries With Lowest Obesity in the World',
             labels={'x': '', 'y': 'Average Obesity (%)'})

        low.update_layout(xaxis={'tickangle': 45, 'tickfont': {'size': 10},
                         'tickmode': 'linear', 'title': ''})
        low.update_layout(width=450, height=450)

        low.update_layout(showlegend=False)

        return low
#calculating obesity for different regions and giving layout dimemnsions
    def regions(self):
        global reg
        data = [go.Scatter(x=obesity_region_pivot.columns,
                   y=obesity_region_pivot.loc[region],
                   name=region) for region in obesity_region_pivot.index]

        layout = go.Layout(
                title='Obesity Prevalence by Region',
                yaxis=dict(title='Obesity Prevalence (%)'),
                xaxis=dict(title='Year'),
                width=450, 
                height=450
                )
        reg = go.Figure(data=data, layout=layout)

        return reg
#giving the dimesions for creating the globe and the command for hovering
    def main_world(self):
        obesity_sex_df = obesity_data.loc[obesity_data.sex == 'Both sexes', :]
        fig = px.choropleth(
            locations=obesity_sex_df.country.astype(str), 
            color=obesity_sex_df.obesity_prev.astype(float), 
            hover_name=obesity_sex_df.country.astype(str), 
            animation_frame=obesity_sex_df.year.astype(int),
            labels={'animation_frame': 'Selected Year', 'color': 'Obesity Intensity', 'locations': 'Country'}, 
            color_continuous_scale=px.colors.sequential.Blues,
            range_color=[0,50],
            locationmode='country names',
            height=700,
            width=700,
            projection='orthographic'  
        )
#giving the color for the globe
        fig.update_geos(
            showocean=True, oceancolor="rgb(14, 47, 77)",
            showland=True, landcolor="grey",
            showcountries=True, countrycolor="white",
            showcoastlines=True, coastlinecolor="white",
            showframe=False
        )
        fig.update_coloraxes(colorbar=dict(xpad=30))
        return fig  
 #calculating obesity for different countries    
    def world(self):
        obesity_sex_df = obesity_data.loc[obesity_data.sex == 'Both sexes', :]
#creating a choropleth map with different color luminosity
        fig = px.choropleth(
            locations=obesity_sex_df.country.astype(str), 
            color=obesity_sex_df.obesity_prev.astype(float), 
            hover_name=obesity_sex_df.country.astype(str), 
            animation_frame=obesity_sex_df.year.astype(int),
            labels={'animation_frame': 'Selected Year', 'color': 'Obesity Intensity', 'locations': 'Country'}, 
            color_continuous_scale=px.colors.diverging.Tealrose,
            range_color=[0,50],
            locationmode='country names',
            height=700,
            width=700,
            title='Global Obesity Indicator',
            projection='orthographic'  
            )
        return fig
        
    def male(self):
#calculating male obesity
        obesity_male_df = obesity_data.loc[obesity_data.sex == 'Male', :]

        fig = px.choropleth(
            locations=obesity_male_df.country.astype(str), 
            color=obesity_male_df.obesity_prev.astype(float), 
            hover_name=obesity_male_df.country.astype(str), 
            animation_frame=obesity_male_df.year.astype(int),
            labels={'animation_frame': 'Selected Year', 'color': 'Obesity Intensity', 'locations': 'Country'}, 
            color_continuous_scale=px.colors.sequential.Blues,
            range_color=[0,50],
            locationmode='country names',
            height=700,
            width=700,
            title='Global Male Obesity Indicator',
            projection='orthographic'  
            )
        return fig
    #calculating female obesity
    def female(self):

        obesity_female_df = obesity_data.loc[obesity_data.sex == 'Female', :]

        fig = px.choropleth(
            locations=obesity_female_df.country.astype(str), 
            color=obesity_female_df.obesity_prev.astype(float), 
            hover_name=obesity_female_df.country.astype(str), 
            animation_frame=obesity_female_df.year.astype(int),
            labels={'animation_frame': 'Selected Year', 'color': 'Obesity Intensity', 'locations': 'Country'}, 
            color_continuous_scale=px.colors.sequential.Burg,
            range_color=[0,50],
            locationmode='country names',
            height=700,
            width=700,
            title='Global Female Obesity Indicator',
            projection='orthographic'  
            )
        return fig
    
