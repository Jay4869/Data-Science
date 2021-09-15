#!/bin/bash
cd /d/Projects/COVID-19
git pull
cd /d/Projects/Data-Science/Projects/COVID-19/
python Data-Processing.py
git add covid_19_US.csv
git add covid_states.csv
git add Time_series_confirmed_cases.csv
git add Time_series_confirmed_cases_states.csv
git commit -m 'JHU daily refresh'
git push
