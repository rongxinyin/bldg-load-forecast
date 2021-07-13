import pandas as pd
import numpy as np
import os
import pathlib

from datetime import timedelta, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn import linear_model

class CleanSiteData(object):
    def __init__(self, site):
        self.site = site
        self.root_dir = pathlib.Path.cwd()
        self.data_dir = self.root_dir.joinpath('data')
        self.out_dir = self.root_dir.joinpath('output/{}'.format(site))
        self.plot_dir = self.root_dir.joinpath('plot/{}'.format(site))

        # Create directories
        for dir_inst in [self.out_dir, self.plot_dir]:
            try: 
                if not os.path.exists(dir_inst):
                    os.makedirs(dir_inst)
            except FileExistsError:
                continue
        # read meter, weather and event data
        self.holidays = self.read_special_days()
        self.event_days = self.read_event_days()
        self.data = self.merge_meter_weather() # used for weather regression model

    def read_special_days(self):
        holidays = pd.read_csv(
            self.root_dir.joinpath('holiday.csv'), usecols=[0])
        holidays['date'] = pd.to_datetime(
            pd.Series(holidays['date']), format='%m/%d/%y')
        holidays['day'] = holidays.date.apply(lambda x: x.strftime('%Y-%m-%d'))
        return holidays

    def read_event_days(self):
        dr_event = pd.read_csv(self.root_dir.joinpath('dr-event-new.csv'))
        dr_event['event_date'] = dr_event.event_date.apply(
            lambda x: datetime.strptime(x, '%m/%d/%y'))
        dr_event['date'] = dr_event.event_date.apply(
            lambda x: x.strftime('%Y-%m-%d'))
        return dr_event

    def read_meter_data(self):
        df = pd.read_csv(self.data_dir.joinpath(
            'meter-data/{}.csv'.format(self.site['site_id'])), index_col=[0], parse_dates=True)
        # df.rename(columns={'_c3': 'power'}, inplace=True)
        df = df.fillna(method='ffill')
        # remove duplicated index
        df = df[~df.index.duplicated(keep='first')]

        # clean dataframe
        df['date'] = df.index
        # df['year'] = df.date.apply(lambda x: x.strftime('%Y'))
        df['time'] = df.date.apply(lambda x: x.strftime('%H:%M'))
        # df['month'] = df.date.apply(lambda x: int(x.strftime('%m')))
        df['day'] = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
        df['hour'] = df.date.apply(lambda x: int(x.strftime('%H')))
        df['weekday'] = df.date.apply(lambda x: int(x.strftime('%w')))
        df['DR'] = df.date.apply(
            lambda x: x.strftime('%Y-%m-%d') in self.event_days[self.event_days['site'] == self.site['site_id']].date.values)
        df['DR'] = df.DR.astype(int)
        df['holiday'] = df.date.apply(lambda x: x.strftime(
            '%Y-%m-%d') in self.holidays.day.values)
        df['holiday'] = df.holiday.astype(int)
        df['valid_dates'] = 0
        df.loc[(df['weekday'] > 0) & (df['weekday'] < 6) & (
            df['holiday'] == 0) & (df['DR'] == 0), ['valid_dates']] = 1

        print('read the meter data.')

        return df

    def read_weather_data(self):
        df = pd.read_csv(self.data_dir.joinpath('weatherdata/{}_2018-01-01_2020-01-01_Weather.csv'.format('SanBernadino')),
                         index_col=[0], parse_dates=True)
        # df.index = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S')
        df['oat'] = df.temperature*1.8+32
        df = df.fillna(method='ffill')
        df = df[~df.index.duplicated(keep='last')]

        # resample to 15 minutes
        df = df.resample('15min').interpolate(method='linear')
        print('read the weather data.')
        # print(df.head())

        return df['oat']

    def merge_meter_weather(self):
        merge_meter_oa = pd.concat(
            [self.read_meter_data(), self.read_weather_data()], axis=1, join='inner')
        return merge_meter_oa

# Avg baseline, Avg baseline with adjustment, matching baseline, OAT regression model
def select_y_baseline_days(input_data, event, y):
    event_date = datetime.strptime(event['event_date'],'%Y-%m-%d')
    # give an empty dataframe to store the baseline day
    baseline_y_data = pd.DataFrame()
    # select 
    i = 1
    j = 0
    # assume can get y baseline days within 20 previous days
    while i < 20:
        baseline_day = datetime.strftime(event_date - timedelta(days=i),
                                        '%Y/%m/%d')
        valid_day = input_data.loc[baseline_day]['valid_dates'].max()
        if valid_day == 1 and j < y:
            baseline_y_data = baseline_y_data.append(input_data.loc[baseline_day], sort=False)
            j = j + 1
        i = i + 1
    return baseline_y_data

def calc_x_y_baseline(baseline_y_data, x):
    # calculate and sort daily peak power and select the highest x days
    selected_x_days = baseline_y_data.groupby(
        'day')['power'].max().sort_values()[-x:].index.tolist()
    selected_x_data = baseline_y_data[baseline_y_data['day'].isin(
        selected_x_days)]
    return selected_x_data

def oat_reg_model(df, event_date):
    params = pd.DataFrame(columns=['day', 'hour', 'a', 'b'])
    for hr in range(24):
        X_hr = df.loc[(df['hour'] == hr), ['oat']]
        y_hr = df.loc[(df['hour'] == hr), ['power']]
        regr = linear_model.LinearRegression()
        regr.fit(X_hr, y_hr)

        # print('Intercept: \n', regr.intercept_)
        # print('Coefficients: \n', regr.coef_)

        # model_param = [regr.intercept_[0], regr.coef_[0][0]]
        params.loc[hr] = [event_date] + [hr, regr.intercept_[0], regr.coef_[0][0]]
    return params

def calc_match_baseline(baseline_data, dr_data):
    # get the matching baseline day from previous 10 valid baseline days
    oa_sum_diff = pd.Series(index=list(set(baseline_data.day)),dtype='float64')
    for i in baseline_data.day.unique():
        #             print(baseline_data[i].oat.values-dr_data.oat.values)
        try:
            oa_sum_diff[i] = np.sum(
                np.square(baseline_data.loc[i].oat.values - dr_data.oat.values), axis=0)
        except ValueError:
            print('Missing OA data on {}'.format(i))
    match_day = oa_sum_diff.idxmin()
    print(match_day)
    try:
        dr_data['match_baseline'] = baseline_data.loc[match_day]['power'].values
    except Exception as e:
        print(e)
    return dr_data[['oat','match_baseline']]

# Calculate model metrics
def calc_model_metrics(y, y_predicted, num_predictor):
    # calculate model metrics
    MAE = "{:.3f}".format(mean_absolute_error(y, y_predicted))
    nMAE = "{:.3f}".format(mean_absolute_error(y, y_predicted)/np.mean(y))
    MBE = "{:.3f}".format(np.mean(y_predicted-y))
    nMBE = "{:.3f}".format(np.sum(y_predicted-y)/(len(y)-1)/np.mean(y))
    RSME = "{:.3f}".format(sqrt(mean_squared_error(y, y_predicted)))
    nRSME = "{:.3f}".format(
        sqrt(mean_squared_error(y, y_predicted))/np.mean(y))
    R2 = "{:.3f}".format(r2_score(y, y_predicted))
    R2_adj = 1 - (1 - r2_score(y, y_predicted)) * \
        ((y.shape[0] - 1) / (y.shape[0] - num_predictor - 1))
    return [MAE, nMAE, MBE, nMBE, RSME, nRSME, R2, R2_adj]
    