import numpy as np
import pandas as pd
import datetime
import os


def process(file):

    # read files
    temp = pd.read_csv(file)
    # filter US country
    temp = temp[temp['Country/Region'] == 'US'] if 'Country/Region' in temp.columns else temp[temp['Country_Region'] == 'US']

    # State & County
    if 'Admin2' in temp.columns:
        temp['County'] = temp.Admin2
        temp['State'] = temp.Province_State
    elif ',' in 'Province/State':
        x = temp['Province/State'].map(
            lambda x: x.replace(' (From Diamond Princess)', '').replace('Unassigned Location', '').split(','))
        temp['County'] = x[0]
        temp['State'] = x[-1]
    else:
        temp['County'] = ''
        temp['State'] = temp['Province/State'].map(
            lambda x: x.replace(' (From Diamond Princess)', '').replace('Unassigned Location', ''))

    # Date
    if 'Last Update' in temp.columns:
        temp['Date'] = pd.to_datetime(temp['Last Update']).map(lambda x: x.strftime('%Y-%m-%d'))
    else:
        temp['Date'] = pd.to_datetime(temp['Last_Update']).map(lambda x: x.strftime('%Y-%m-%d'))

    # Geograph
    temp['Lat'] = temp['Latitude'] if 'Latitude' in temp.columns else np.nan

    if 'Longitude' in temp.columns:
        temp['Long'] = temp['Longitude']
    elif 'Long_' in temp.columns:
        temp['Long'] = temp['Long_']
    else:
        temp['Long'] = np.nan

    temp['Deaths'] = temp.Deaths.fillna(0, inplace=True)
    temp['Recovered'] = temp.Recovered.fillna(0, inplace=True) if 'Recovered' in temp.columns else 0
    temp['Active'] = temp['Active'] if 'Active' in temp.columns else 0
    temp['FIPS'] = temp['FIPS'] if 'FIPS' in temp.columns else 0

    temp = temp[['FIPS', 'County', 'State', 'Date', 'Lat', 'Long', 'Confirmed', 'Deaths', 'Recovered', 'Active']]

    return temp

if __name__ == '__main__':
    path = '../../Data/covid_19_daily_reports/'

    # Loading all files, or updating the latest files
    if not os.path.exists(path + 'covid_19_US.csv'):
        files = [i for i in os.listdir(path) if 'csv' in i]
        data = pd.DataFrame([])

        for i in files:
            print('loading {}'.format(i))
            temp = process(path + i)
            data = pd.concat([data, temp], ignore_index=True)

        print(data.shape)
    else:
        data = pd.read_csv(path + 'covid_19_US.csv').drop('Unnamed: 0', axis=1)
        files = [i for i in os.listdir(path) if i[:5] > max(data.Date)[-5:] and '2020' in i]

        for i in files:
            print('loading {}'.format(i))

            try:
                temp = process(path + i)
            except ValueError:
                print('New file has different format.')

            assert (data.shape[1] == 10)
            data = pd.concat([data, temp], ignore_index=True)

            print('{0} new records have been loaded.'.format(temp.shape[0]))

    print('There are total {} records.'.format(data.shape[0]))

    data.to_csv(path + 'covid_19_US.csv')
