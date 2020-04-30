import numpy as np
import pandas as pd
import datetime
import os

def ETL(file):

    # loading COVID-19 data and zipcode
    print('Processing {}'.format(file[-14:]))
    data = pd.read_csv(file)
    data = data.loc[data.Country_Region == 'US', ['FIPS', 'Admin2', 'Province_State', 'Confirmed', 'Deaths']]

    zipinfo = pd.read_csv('uszips.csv')[['county_fips', 'county_name', 'state_name', 'population']].groupby(['county_fips', 'county_name', 'state_name']).population.agg('sum').reset_index()

    # rename features and change format
    data.columns = ['FIPS', 'County', 'State', 'Confirmed', 'Deaths']
    data.FIPS.fillna(0, inplace=True)
    data.FIPS = data.FIPS.astype(int).astype(str).map(lambda x: '0'+x if len(x) == 4 and x != '0000' else x)    # remove decimal

    zipinfo.columns = ['FIPS', 'County', 'State', 'population']
    zipinfo.FIPS = zipinfo.FIPS.astype(str).map(lambda x: '0'+x if len(x) == 4 and x != '0000' else x)

    # remove 'City' in the County columns to match zipcode table
    data.County = data.County.str.replace(' City', '', regex=False)

    # final output features
    feature = ['FIPS', 'County', 'State', 'population', 'Confirmed', 'Deaths']

    # extract non-missing records
    x = pd.merge(data, zipinfo, on=['FIPS', 'County', 'State'], how='inner')[feature]

    # correct wrong FIPS and fill in missing
    y = data[~data.FIPS.isin(x.FIPS)]
    z = pd.merge(y[['County', 'State', 'Confirmed', 'Deaths']], zipinfo, on=['County', 'State'], how='inner')[feature]

    # merge two tables and sum same county records
    data = pd.concat([x, z], ignore_index=True).groupby(['FIPS', 'County', 'State', 'population']).agg('sum').reset_index()

    return data

def increment(x, y):

    # merge two current and previous dataset
    temp = pd.merge(x, y, on=['FIPS', 'County', 'State', 'population'], how='left')

    # fill NA for new county records
    temp.Confirmed_y.fillna(0, inplace=True)
    temp.Deaths_y.fillna(0, inplace=True)

    # calculate incremental increase
    temp['Confirmed_new'] = temp.Confirmed_x - temp.Confirmed_y
    temp['Deaths_new'] = temp.Deaths_x - temp.Deaths_y
    temp['Confirmed'] = temp.Confirmed_x
    temp['Deaths'] = temp.Deaths_x
    temp['Confirmed_rate'] = temp.Confirmed / temp.population
    temp['Deaths_rate'] = temp.Deaths / temp.Confirmed

    # final output
    feature = ['FIPS', 'County', 'State', 'Confirmed', 'Confirmed_new', 'Confirmed_rate', 'Deaths', 'Deaths_new', 'Deaths_rate']

    # remove useless features and save CSV
    temp[feature].to_excel('covid_19_US.xlsx', index=False)
    print(temp[feature].shape)

if __name__ == '__main__':
    path = '../../../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'

    file = [i for i in os.listdir(path) if 'csv' in i]
    current = ETL(path + file[-1])
    previous = ETL(path + file[-2])
    increment(current, previous)