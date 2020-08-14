import numpy as np
import pandas as pd
import datetime
import os
import us

def summary(file):

    # loading COVID-19 data and zipcode
    print('Processing {}'.format(file[-14:]))
    data = pd.read_csv(file).query('Country_Region == "US" and Province_State != "Recovered"')[['FIPS', 'Admin2', 'Province_State', 'Confirmed', 'Deaths', 'Active']]

    # remove 'City' in the County columns to match zipcode table
    data.Admin2 = data.Admin2.str.replace(' City', '', regex=False)
    zipinfo.Admin2 = zipinfo.Admin2.str.replace(' City', '', regex=False)

    # final output features
    feature = ['FIPS', 'County', 'State_id', 'State', 'Population', 'Confirmed', 'Deaths', 'Active']

    # extract non-missing records
    temp = pd.merge(zipinfo, data, on=['FIPS', 'Admin2', 'Province_State'], how='inner')
    temp.insert(2, 'State_id', temp.Province_State.map(us.states.mapping('name', 'abbr')))
    temp.columns = feature

    return temp

def increment(x, y):

    # merge two current and previous dataset
    temp = pd.merge(x, y, on=['FIPS', 'County', 'State_id', 'State', 'Population'], how='left')

    # fill NA for new county records
    temp.Confirmed_y.fillna(0, inplace=True)
    temp.Deaths_y.fillna(0, inplace=True)

    # calculate incremental increase
    temp.eval('Confirmed_new = Confirmed_x - Confirmed_y', inplace=True)
    temp.eval('Deaths_new = Deaths_x - Deaths_y', inplace=True)
    temp.eval('Confirmed = Confirmed_x', inplace=True)
    temp.eval('Deaths = Deaths_x', inplace=True)
    temp.eval('Active = Active_x', inplace=True)
    temp.eval('Confirmed_per_1000 = Confirmed_x / Population * 1000', inplace=True)
    temp.eval('Deaths_per_1000 = Deaths_x / Population * 1000', inplace=True)
    temp.eval('Actives_per_1000 = Active_x / Population * 1000', inplace=True)

    # final output
    feature = ['FIPS', 'County', 'State_id', 'State', 'Population', 'Confirmed', 'Deaths', 'Active', 'Confirmed_new', 'Deaths_new', 'Confirmed_per_1000', 'Deaths_per_1000', 'Actives_per_1000']

    # remove useless features and save CSV
    temp[feature].to_excel('covid_19_US.xlsx', index=False)
    print('Finish ETL: county-level summary, and total confirmed cases:', temp.Confirmed.sum())

def ts(file):
    
    # load US national time series data
    data = pd.read_csv(file)
    feature = [i for i in data.columns if '20' in i]

    # aggregate top 10 confirmed cases states
    top10 = data.groupby('Province_State')[feature].agg('sum').sum(axis=1).sort_values(ascending=False).head(10).keys()
    data.groupby('Province_State')[feature].agg('sum').transpose()[top10].to_csv('Top_10_states.csv', index=False)
    
    # aggregate daily cases
    ts = data[feature].sum(axis=0)
    ts.index = pd.to_datetime(ts.index)
    ts = (ts - ts.shift(1)).fillna(0).reset_index()
    ts.columns = ['Date', 'Confirmed']
    
    ts.to_csv('Time_series_confirmed_cases.csv', index=False)
    print('Finish ETL: time-series cases, and latest reported:', ts.Confirmed.values[-1])
    
    return ts
    
if __name__ == '__main__':
    
    # set path
    path = 'd:/Projects/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'
    file = [i for i in os.listdir(path) if 'csv' in i]
    
    # load zipcode info
    zipinfo = pd.read_csv(path + '../UID_ISO_FIPS_LookUp_Table.csv')[['FIPS', 'Admin2', 'Province_State', 'Population']]
    
    # county-level summary
    current = summary(path + file[-1])
    previous = summary(path + file[-2])
    increment(current, previous)
    
    # time series
    file = 'd:/Projects/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    confirmed_ts = ts(file)
    
    