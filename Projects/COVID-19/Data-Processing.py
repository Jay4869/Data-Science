import numpy as np
import pandas as pd
import datetime
import os
import us

def summary(file):

    # loading COVID-19 data and zipcode
    print('Processing {}'.format(file[-14:]))
    data = pd.read_csv(file).query('Country_Region == "US" and Province_State not in ("Diamond Princess", "Grand Princess", "Recovered")')[['FIPS', 'Admin2', 'Province_State', 'Confirmed', 'Deaths', 'Active']].dropna(how='all')

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
    temp.eval('Confirmed_per_100k = Confirmed_new / Population * 100000', inplace=True)
    temp.eval('Deaths_per_100k_confirmed = Deaths_x / Confirmed * 100000', inplace=True)
    temp.eval('Actives_per_100k = Active_x / Population * 100000', inplace=True)

    # final output
    feature = ['FIPS', 'County', 'State_id', 'State', 'Population', 'Confirmed', 'Deaths', 'Active', 'Confirmed_new', 'Deaths_new', 'Confirmed_per_100k', 'Deaths_per_100k_confirmed', 'Actives_per_100k']

    # remove useless features and save CSV
    temp[feature].to_csv('covid_19_US.csv', index=False)
    print('Finish ETL: county-level summary, and total confirmed cases:', temp.Confirmed.sum())
    print('New cases are reported in LA:', temp.query('County == "Los Angeles"').Confirmed_new.sum())
    
    # state-level
    temp = temp.groupby(['State_id', 'State'])[['State_id', 'State', 'Population', 'Confirmed', 'Deaths', 'Active', 'Confirmed_new']].agg('sum')
    temp.eval('Confirmed_per_100k = Confirmed_new / Population * 100000', inplace=True)
    temp.eval('Deaths_per_100k_confirmed = Deaths / Confirmed * 100000', inplace=True)
    temp.eval('Actives_per_100k = Active / Population * 100000', inplace=True)

    temp.to_csv('covid_states.csv')
    print('Finish ETL: state-level summary, and yesterday reported in CA:', temp.query('State_id == "CA"').Confirmed_new.values[0])
    
def ts(file):
    
    # load US national time series data
    data = pd.read_csv(file).dropna(how='all')
    feature = [i for i in data.columns if '20' in i]

    # aggregate confirmed cases by states
    temp = data.groupby('Province_State')[feature].agg('sum').transpose()
    temp.index = pd.to_datetime(temp.index)
    temp.to_csv('Time_series_confirmed_cases_states.csv')
    print('Finish ETL: time-series cases by state.')
    
    # aggregate daily cases
    ts = data[feature].sum(axis=0)
    ts.index = pd.to_datetime(ts.index)
    ts = (ts - ts.shift(1)).fillna(0).reset_index()
    ts.columns = ['Date', 'Confirmed']
    
    ts.to_csv('Time_series_confirmed_cases.csv', index=False)
    print('Finish ETL: time-series cases, and yesterday reported:', ts.Confirmed.values[-1])
    
    return ts
    
if __name__ == '__main__':
    
    # set path
    path = 'd:/Projects/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/'
    file = [i for i in os.listdir(path) if 'csv' in i]
    
    # load zipcode info
    zipinfo = pd.read_csv(path + '../UID_ISO_FIPS_LookUp_Table.csv').query('Province_State not in ("Recovered", "Guam", "Diamond Princess", "Grand Princess")')[['FIPS', 'Admin2', 'Province_State', 'Population']]
    
    # county-level summary
    current = summary(path + file[-1])
    previous = summary(path + file[-2])
    increment(current, previous)
    
    # time series
    file = 'd:/Projects/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    confirmed_ts = ts(file)
    
    