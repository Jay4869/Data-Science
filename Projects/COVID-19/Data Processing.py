import numpy as np
import pandas as pd
import datetime
import os

def ETL(file):

    # loading COVID-19 data and zipcode
    print('Processing {}'.format(file[-14:]))
    data = pd.read_csv(file)
    zipinfo = pd.read_csv('uszips.csv')[['county_fips', 'county_name', 'state_name', 'lat', 'lng']].drop_duplicates()
    zipinfo = zipinfo.groupby(['county_fips', 'county_name', 'state_name']).agg('mean').reset_index()

    # extract US records and features
    data = data.loc[data.Country_Region == 'US', ['FIPS', 'Admin2', 'Province_State', 'Confirmed', 'Deaths', 'Recovered', 'Active']]

    # rename features
    data.columns = ['FIPS', 'County', 'State', 'Confirmed', 'Deaths', 'Recovered', 'Active']

    # remove 'City' in the County columns to match zipcode table
    data.County = data.County.str.replace(' City', '', regex=False)

    # final output features
    feature = ['FIPS', 'County', 'State', 'lat', 'lng', 'Confirmed', 'Deaths', 'Recovered', 'Active']

    # extract non-missing records
    x = pd.merge(data, zipinfo, left_on = ['FIPS', 'County', 'State'],
                 right_on = ['county_fips', 'county_name', 'state_name'], how = 'inner')[feature]

    # correct wrong FIPS and fill in missing
    y = data[~data.FIPS.isin(x.FIPS)]
    z = pd.merge(y, zipinfo, left_on = ['County', 'State'], right_on = ['county_name', 'state_name'], how = 'inner')
    z.FIPS = z.county_fips
    z = z[feature]

    # merge two tables
    data = pd.concat([x, z], ignore_index = True).groupby(['FIPS', 'County', 'State']).agg('sum').reset_index()

    return data

def increment(x, y):
    temp = pd.merge(x, y[['FIPS', 'County', 'State', 'Confirmed', 'Deaths', 'Recovered', 'Active']],
                    on=['FIPS', 'County', 'State'], how='left')

    temp['Confirmed_new'] = temp.Confirmed_x - temp.Confirmed_y
    temp['Deaths_new'] = temp.Deaths_x - temp.Deaths_y
    # temp['Recovered_new'] = temp.Recovered_x - temp.Recovered_y
    temp['Confirmed'] = temp.Confirmed_x
    temp['Deaths'] = temp.Deaths_x
    # temp['Recovered'] = temp.Recovered_x
    temp['Active'] = temp.Active_x
    temp = temp[['FIPS', 'County', 'State', 'lat', 'lng', 'Confirmed', 'Confirmed_new',
              'Deaths', 'Deaths_new', 'Active']]

    pop = pd.read_csv('us_county.csv')[['fips', 'population']]
    temp = pd.merge(temp, pop, left_on='FIPS', right_on='fips', how='left').drop('fips', axis=1)
    temp.eval("Confirmed_rate = Confirmed / population", inplace=True)
    temp.eval("Deaths_rate = Deaths / population", inplace=True)

    temp.drop('population', axis=1, inplace=True)
    temp.to_csv('covid_19_US.csv', index=False)
    print(temp.shape)

if __name__ == '__main__':
    path = '../../../COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'

    file = [i for i in os.listdir(path) if 'csv' in i]
    current = ETL(path + file[-1])
    previous = ETL(path + file[-2])
    increment(current, previous)