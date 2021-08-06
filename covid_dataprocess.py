import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

DEBUG = False

class COVID_dataprocess():
    def __init__(self,
                 path,
                 source_type,
                 train_state_list,
                 test_state_list,
                 if_onlynewcases,
                 if_moving_avg_data,
                 if_log_scale,
                 input_variable_list = []):

        self.path = path
        self.source_type = source_type
        self.train_state_list = train_state_list
        self.test_state_list = test_state_list
        self.if_onlynewcases = if_onlynewcases
        self.if_moving_avg_data = if_moving_avg_data
        self.if_log_scale = if_log_scale
        self.input_variable_list = input_variable_list
        self.result_dict = {}


        if self.source_type == 'COVID_ACT_NOW':
            print('LOADING.. COVID_ACT_NOW')
            self.input_variable_list = ['input_hospitalBedsRequired',
                                        'input_ICUBedsInUse',
                                        'input_ventilatorsInUse',
                                        'input_Deaths',
                                        'input_Infected']
            self.df = self.read_data_COVIDactnow_NYT()
            self.df = self.truncate_NAN_DataFrame()
            self.df = self.moving_avg_log_scale()
            self.data_test = self.extract_ndarray_from_DataFrame()
            self.result_dict.update({'Data source': 'COVID_ACT_NOW'})
            self.result_dict.update({'Full DataFrame': self.df})
            self.result_dict.update({'Data array': self.data_test})
            self.result_dict.update({'List_states': self.train_state_list})
            self.result_dict.update({'List_variables': self.input_variable_list})



        elif self.source_type == 'COVID_TRACKING_PROJECT':
            print('LOADING.. COVID_TRACKING_PROJECT')
            self.input_variable_list = ['input_hospitalized_Currently',
                                        'input_inICU_Currently',
                                        'input_daily_test_positive_rate',
                                        'input_daily_cases',
                                        'input_daily_deaths']
            # 'input_daily_cases_pct_change']

            self.df = self.read_data_COVIDtrackingProject()
            self.df = self.truncate_NAN_DataFrame()
            self.df = self.moving_avg_log_scale()
            self.extract_ndarray_from_DataFrame()

            self.result_dict.update({'Full DataFrame': self.df})
            self.result_dict.update({'Data array (test)': self.data_test}) # state x days x variables
            self.result_dict.update({'List_states (test)': self.test_state_list})
            self.result_dict.update({'List_variables': self.input_variable_list})
            self.result_dict.update({'Data array (train)': self.data_train}) # state x days x variables
            self.result_dict.update({'List_states (train)': self.train_state_list})


        else:  ### JHU data
            print('LOADING.. JHU Data')
            self.data_test = self.combine_data(self.source)


    def moving_avg_log_scale(self):
        df = self.df
        state_list_combined = self.train_state_list + self.test_state_list
        state_list_combined = list(set(state_list_combined))

        if self.if_moving_avg_data:
            for state in state_list_combined:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = df2.rolling(window=5, win_type=None).sum() / 5  ### moving average with backward window size 5
                df2 = df2.fillna(0)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        if self.if_log_scale:
            for state in state_list_combined:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = np.log(df2 + 1)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        return df

    def truncate_NAN_DataFrame(self):
        df = self.df.copy()
        ### Take the maximal sub-dataframe that does not contain NAN
        ### If some state has all NANs for some variable, that variable is dropped from input_list_variable
        start_dates = []
        end_dates = []
        state_list_combined = self.train_state_list + self.test_state_list
        state_list_combined = list(set(state_list_combined))
        print('!!! self.train_state_list', self.train_state_list)
        print('!!! state_list_combined', state_list_combined)

        input_variable_list_noNAN = self.input_variable_list.copy()
        for column in input_variable_list_noNAN:
            for state in state_list_combined:
                df1 = df.get(state)

                if df1[column].isnull().all():
                    input_variable_list_noNAN.remove(column)
        self.input_variable_list = input_variable_list_noNAN
        print('!!! New input_variable_list', self.input_variable_list)

        for state in state_list_combined:
            df1 = df.get(state)
            for column in self.input_variable_list:
                l_min = df1[column][df1[column].notnull()].index[0]
                l_max = df1[column][df1[column].notnull()].index[-1]
                start_dates.append(l_min)
                end_dates.append(l_max)

        max_min_date = max(start_dates)
        min_max_date = min(end_dates)

        for state in state_list_combined:
            df1 = df.get(state)
            df1 = df1[max_min_date:min_max_date]
            print('!!! If any value is NAN:', df1.isnull())
            df.update({state: df1})
        return df

    def extract_ndarray_from_DataFrame(self):
        ## Make numpy array of shape States x Days x variables for each test and train sets
        data_combined = []
        df = self.df
        data_test = []
        data_train = []

        print('!!! self.state_list', self.test_state_list)
        for state in self.test_state_list:
            df1 = df.get(state)
            data_combined = df1[self.input_variable_list].values  ## shape Days x variables
            data_test.append(data_combined)

        for state in self.train_state_list:
            df2 = df.get(state)
            data_combined = df2[self.input_variable_list].values  ## shape Days x variables
            data_train.append(data_combined)

        data_test = np.asarray(data_test)
        self.data_test = np.nan_to_num(data_test, copy=True, nan=0, posinf=1, neginf=0)
        print('!!!! data_test.shape', data_test.shape)
        data_train = np.asarray(data_train)
        self.data_train = np.nan_to_num(data_train, copy=True, nan=0, posinf=1, neginf=0)
        print('!!!! data_train.shape', data_train.shape)


    """
    def read_data_COVIDtrackingProject(self):
        '''
        Read input time series data as a dictionary of pandas dataframe
        '''
        print('??? Loading.. read_data_COVIDtrackingProject')


        data = pd.read_csv(self.path, delimiter=',').sort_values(by="date")
        ### Convert the format of dates from string to datetime
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', utc=False)

        df = {}

        state_list_full = sorted([i for i in set([i for i in data['state']])])
        print('!!! state_list_full', state_list_full)


        ### Find earliest starting date of the data
        start_dates = []
        for state in state_list_full:
            df1 = data.loc[data['state'] == state]
            start_dates.append(min(df1['date']).strftime("%Y-%m-%d"))
        max_min_date = max(start_dates)
        print('!!! min_dates', max_min_date)

        for state in state_list_full:
            df1 = data.loc[data['state'] == state].set_index('date')
            # lastUpdatedDate = df1['lastUpdateEt'].iloc[0]
            df1 = df1[max_min_date:]
            ### making new columns to process columns of interest and preserve the original data
            df1['input_onVentilator_Increase'] = df1['onVentilatorCumulative']
            df1['input_inICU_Increase'] = df1['inIcuCumulative']
            df1['input_test_positive_rate'] = df1['positiveTestsViral'] / df1['totalTestsViral']
            df1['input_case_Increase'] = df1['positiveIncrease']
            df1['input_death_Increase'] = df1['deathIncrease']

            df.update({state: df1})

        if self.if_moving_avg_data:
            for state in state_list_full:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = df2.rolling(window=5, win_type=None).sum() / 5  ### moving average with backward window size 5
                df2 = df2.fillna(0)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        if self.if_log_scale:
            for state in state_list_full:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = np.log(df2 + 1)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        self.df = df


        ## Make numpy array of shape States x Days x variables
        data_combined = []
        for state in state_list_full:
            df1 = df.get(state)

            if state == state_list_full[0]:
                data_combined = df1[self.input_variable_list].values  ## shape Days x variables
                data_combined = np.expand_dims(data_combined, axis=0)
                print('!!!Data_combined.shape', data_combined.shape)
            else:
                data_new = df1[self.input_variable_list].values  ## shape Days x variables
                data_new = np.expand_dims(data_new, axis=0)
                print('!!! Data_new.shape', data_new.shape)
                data_combined = np.append(data_combined, data_new, axis=0)

        self.data_test = data_combined
    """


    def combine_data(self, source):
        if len(source) == 1:
            for path in source:
                data, self.country_list = self.read_data_as_array_countrywise(path)
                data_combined = np.expand_dims(data, axis=2)

        else:
            path = source[0]
            data, self.country_list = self.read_data_as_array_countrywise(path)

            data_combined = np.empty(shape=[data.shape[0], data.shape[1], 1])
            for path in source:
                data_new = self.read_data_as_array_countrywise(path)[0]
                data_new = np.expand_dims(data_new, axis=2)
                # print('data_new.shape', data_new.shape)
                min_length = np.minimum(data_combined.shape[1], data_new.shape[1])
                data_combined = np.append(data_combined[:, 0:min_length, :], data_new[:, 0:min_length, :], axis=2)
            data_combined = data_combined[:, :, 1:]

            print('data_combined.shape', data_combined.shape)
        # data_full.replace(np.nan, 0)  ### replace all NANs with 0

        ### Replace all NANs in data_combined with 0
        where_are_NaNs = np.isnan(data_combined)
        data_combined[where_are_NaNs] = 0
        return data_combined


    def read_data_COVIDtrackingProject(self):
        '''
        Read input time series data as a dictionary of pandas dataframe
        '''


        # path = "Data/us_states_COVID_tracking_project.csv"
        data = pd.read_csv(self.path, delimiter=',').sort_values(by="date")
        ### Convert the format of dates from string to datetime
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', utc=False)

        df = {}

        ### Use full state names
        state_list = sorted([i for i in set([i for i in data['state']])])
        print('!!! state_list', state_list)


        ### Find maximum earliest and the minimum latest date of both data
        start_dates = []
        end_dates = []
        for state in state_list:
            df1 = data.loc[data['state'] == state]
            start_dates.append(min(df1['date']).strftime("%Y-%m-%d"))
            end_dates.append(max(df1['date']).strftime("%Y-%m-%d"))
            # print('State %s and end_date %s' % (state, max(df1['date']).strftime("%Y-%m-%d")))
        max_min_date = max(start_dates)
        min_max_date = min(end_dates)

        print('!!! max_min_date', max_min_date)
        print('!!! min_max_date', min_max_date)


        original_list_variables = data.keys().tolist()
        original_list_variables.remove('date')


        for state in state_list:
            df1 = data.loc[data['state'] == state].set_index('date')
            # lastUpdatedDate = df1['lastUpdateEt'].iloc[0]
            df1 = df1[max_min_date:min_max_date]
            ### making new columns to process columns of interest and preserve the original data
            df1['input_hospitalized_Currently'] = df1['hospitalizedCurrently']
            df1['input_inICU_Currently'] = df1['inIcuCurrently']
            df1['input_daily_test_positive_rate'] = df1['positive'].diff() / df1['totalTestResults'].diff()
            df1['input_daily_cases'] = df1['positive'].diff()
            df1['input_daily_deaths'] = df1['death'].diff()
            df1['input_daily_cases_pct_change'] = df1['positive'].pct_change()

            # print('!!! If any value is NAN: %r for state %s:' % (df1.isnull().values.any(), state))
            df.update({abbrev_us_state[state]: df1})

        """
        for variable in original_list_variables:
            for state in state_list:
                df1 = data.loc[data['state'] == state].set_index('date')
                if not df1[variable].isnull().values.any():
                    df.update({'list_states_observed_' + variable: abbrev_us_state[state]})
        """

        return df






    def read_data_as_array_countrywise(self, path):
        '''
        Read input time series as a narray
        '''
        data_full = pd.read_csv(path, delimiter=',').T
        data = data_full.values[1:, :]
        data = np.delete(data, [1, 2], 0)  # delete lattitue & altitude
        if self.country_list == None:
            country_list = [i for i in set(data[0, :])]
            country_list = sorted(country_list)  # whole countries in alphabetical order
        else:
            country_list = self.country_list

        ### merge data according to country
        data_new = np.zeros(shape=(data.shape[0] - 1, len(country_list)))
        for i in np.arange(len(country_list)):
            idx = np.where(data[0, :] == country_list[i])
            data_sub = data[1:, idx]
            data_sub = data_sub[:, 0, :]
            data_sub = np.sum(data_sub, axis=1)
            data_new[:, i] = data_sub
        data_new = data_new.astype(int)

        if self.country_list == None:
            idx = np.where(data_new[-1, :] > 1000)
            data_new = data_new[:, idx]
            data_new = data_new[:, 0, :]
            # data_new[:,1] = np.zeros(data_new.shape[0])
            print('data_new', data_new)
            country_list = [country_list[idx[0][i]] for i in range(len(idx[0]))]
            print('country_list', country_list)

        if self.if_onlynewcases:
            data_new = np.diff(data_new, axis=0)

        if self.if_moving_avg_data:
            for i in np.arange(5, data_new.T.shape[1]):
                data_new.T[:, i] = (data_new.T[:, i] + data_new.T[:, i - 1] + data_new.T[:, i - 2] + data_new.T[:,
                                                                                                     i - 3] + data_new.T[
                                                                                                              :,
                                                                                                              i - 4]) / 5
                # A_recons[:, i] = (A_recons[:, i] + A_recons[:, i-1]) / 2

        if self.if_log_scale:
            data_new = np.log(data_new + 1)

        return data_new.T, country_list


    def read_data_JHU_countrywise(self):
        '''
        Read input time series as a narray
        '''
        data_full = pd.read_csv(self.path, delimiter=',').T
        data = data_full.values[1:, :]
        data = np.delete(data, [1, 2], 0)  # delete lattitue & altitude
        country_list = [i for i in set(data[0, :])]
        country_list = sorted(country_list)  # whole countries in alphabetical order

        ### merge data according to country
        data_new = np.zeros(shape=(data.shape[0] - 1, len(country_list)))
        for i in np.arange(len(country_list)):
            idx = np.where(data[0, :] == country_list[i])
            data_sub = data[1:, idx]
            data_sub = data_sub[:, 0, :]
            data_sub = np.sum(data_sub, axis=1)
            data_new[:, i] = data_sub
        data_new = data_new.astype(int)

        idx = np.where(data_new[-1, :] > 1000)
        data_new = data_new[:, idx]
        data_new = data_new[:, 0, :]
        # data_new[:,1] = np.zeros(data_new.shape[0])
        print('data_new', data_new)
        country_list = [country_list[idx[0][i]] for i in range(len(idx[0]))]
        print('country_list', country_list)

        data_new = np.diff(data_new, axis=0)

        return data_new.T,

    def read_data_COVIDactnow_NYT(self):
        '''
        Read input time series data as a dictionary of pandas dataframe
        Get Hospotal related data from COVIDACTNOW and cases and deaths from NYT
        '''

        data_ACT = pd.read_csv("Data/states.NO_INTERVENTION.timeseries.csv", delimiter=',')
        data_NYT = pd.read_csv("Data/NYT_us-states.csv", delimiter=',')
        df = {}

        state_list = sorted([i for i in set([i for i in data_ACT['stateName']])])

        ### Find maximum earliest and the minimum latest date of both data
        start_dates = []
        end_dates = []
        for state in state_list:
            df1 = data_ACT.loc[data_ACT['stateName'] == state]
            df2 = data_NYT.loc[data_NYT['state'] == state]
            start_dates.append(min(df1['date']))
            start_dates.append(min(df2['date']))
            end_dates.append(max(df1['date']))
            end_dates.append(max(df2['date']))

        max_min_date = max(start_dates)
        min_max_date = min(end_dates)
        # print('!!! min_dates', max_min_date)

        for state in state_list:
            df1 = data_ACT.loc[data_ACT['stateName'] == state].set_index('date')
            lastUpdatedDate = df1['lastUpdatedDate'].iloc[0]
            df1 = df1[max_min_date:min(lastUpdatedDate, min_max_date)]
            df1['input_hospitalBedsRequired'] = df1['hospitalBedsRequired']
            df1['input_ICUBedsInUse'] = df1['ICUBedsInUse']
            df1['input_ventilatorsInUse'] = df1['ventilatorsInUse']

            df2 = data_NYT.loc[data_NYT['state'] == state].set_index('date')
            df1['input_Deaths'] = df2['deaths']
            # print('!!! df_NYT1', df2['deaths'])
            df1['input_Infected'] = df2['cases']
            # print('!!! df_NYT1_cases', df2['cases'])

            df1 = df1.fillna(0)

            print('!!! If any value is NAN:', df1.isnull().values.any())

            df.update({state: df1})

        return df







us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))
