import numpy as np
import pandas as pd
import datetime
import os

def ETL(file):

    # loading COVID-19 data and zipcode
    print('Processing {}'.format(file[-14:]))
    data = pd.read_csv(file)
    uszips = pd.read_csv('uszips.csv')
    Geo = uszips.groupby(['county_fips', 'county_name', 'state_name'])['lat', 'lng'].agg('mean').reset_index()
    Pop = uszips.groupby(['county_fips', 'county_name', 'state_name']).population.agg('sum').reset_index()
    Geo = pd.merge(Geo, Pop, on=['county_fips', 'county_name', 'state_name'], how='inner')
    Geo.columns = ['FIPS', 'County', 'State', 'lat', 'lng', 'population']

    # extract US records and features
    data = data.loc[data.Country_Region == 'US', ['FIPS', 'Admin2', 'Province_State', 'Confirmed', 'Deaths']]

    # rename features
    data.columns = ['FIPS', 'County', 'State', 'Confirmed', 'Deaths']

    # remove 'City' in the County columns to match zipcode table
    data.County = data.County.str.replace(' City', '', regex=False)

    # final output features
    feature = ['FIPS', 'County', 'State', 'Confirmed', 'Deaths']

    # extract non-missing records
    x = pd.merge(data, Geo, on=['FIPS', 'County', 'State'], how='inner')[feature]

    # correct wrong FIPS and fill in missing
    y = data[~data.FIPS.isin(x.FIPS)]
    z = pd.merge(y, Geo, on=['County', 'State'], how='inner')
    z['FIPS'] = z.FIPS_y
    z = z[feature]

    # merge two tables and sum same county records
    data = pd.concat([x, z], ignore_index=True).groupby(['FIPS', 'County', 'State']).agg('sum').reset_index()

    # merge Geo info
    data = pd.merge(data, Geo, on=['FIPS', 'County', 'State'], how='inner')

    return data

def increment(x, y):

    # merge two current and previous dataset
    temp = pd.merge(x, y[['FIPS', 'County', 'State', 'Confirmed', 'Deaths']], on=['FIPS', 'County', 'State'], how='left')

    # fill NA for new county records
    temp.Confirmed_y.fillna(0, inplace=True)
    temp.Deaths_y.fillna(0, inplace=True)

    # calculate incremental increase
    temp['Confirmed_new'] = temp.Confirmed_x - temp.Confirmed_y
    temp['Deaths_new'] = temp.Deaths_x - temp.Deaths_y
    temp['Confirmed'] = temp.Confirmed_x
    temp['Deaths'] = temp.Deaths_x

    # calucate confirmed and Death rate
    temp.eval("Confirmed_rate = Confirmed / population", inplace=True)
    temp.eval("Deaths_rate = Deaths / Confirmed", inplace=True)

    # remove useless features and save CSV
    temp = temp[['FIPS', 'County', 'State', 'lat', 'lng',
          'Confirmed', 'Confirmed_new', 'Deaths', 'Deaths_new', 'Confirmed_rate', 'Deaths_rate']]
    temp.to_csv('covid_19_US.csv', index=False)
    print(temp.shape)

if __name__ == '__main__':
    path = '../../../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'

    file = [i for i in os.listdir(path) if 'csv' in i]
    current = ETL(path + file[-1])
    previous = ETL(path + file[-2])
    increment(current, previous)