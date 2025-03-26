"""
Copyright (c) 2025 Tobias Brudermueller, ETH Zurich

This software is released under the MIT License.
See the LICENSE file in the repository root for full license text.
"""

import os 
import pandas as pd 
import numpy as np
import warnings
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

####################################
# BASIC COLORS
####################################

COLOR_LIGHT = '#c2d0d4'
COLOR_DARK = '#35626f'
COLOR_RED = '#E55451'
COLOR_PURPLE = cm.viridis_r(1.0)
COLOR_YELLOW = cm.cividis(0.70)

####################################
# GENERAL HELPER FUNCTIONS
####################################

def mkdir(path):
    '''
        Checks for existence of a path and creates it if necessary: 
        Args: 
            path: (string) path to folder
    '''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError as e:
            if not os.path.exists(path):
                raise

####################################
# CLASS FOR LOADING THE HEAPO DATA 
####################################

class HEAPO():
    '''
        HEAPO: Open Dataset for Heat Pump Optimization with Smart Electricity Meter Data and On-Site Inspection Protocols
    '''
    def __init__(self, data_path=None, use_local_time:bool=False, suppress_warning:bool=False):
        '''
            Provides a class for loading the HEAPO data set.
            Args: 
                data_path: string to root directory, where data set files are located (of the named data set). 
                    NOTE: assuming that the files were unzipped, but not renamed after download
                    NOTE: if parameter is None, assuming that the files are located under a subfolder named "heapo_data" in this repo
                use_local_time: boolean to indicate if the time whenever loading data, timestamp should be converted to local time (Europe/Zurich)
                suppress_warning: boolean to indicate if warnings should be suppressed
        '''

        # handle data path
        assert isinstance(data_path, str) or data_path is None, 'HEAPO.__init__(): Parameter data_path must be a string or None.'
        if isinstance(data_path, str):
            if not data_path.endswith('/'): 
                data_path += '/'
            self.data_path = data_path
        else: 
            self.data_path = os.path.dirname(os.path.abspath(__file__))
            if '\\' in self.data_path: 
                self.data_path = self.data_path.split('\\src')[0]+'/heapo_data/'
            else: 
                self.data_path = self.data_path.split('/src')[0]+'/heapo_data/'
        assert os.path.isdir(self.data_path), 'HEAPO.__init__(): Data path does not exist: {}'.format(self.data_path)

        # define keywords used as reference across the files 
        self.weather_id = 'Weather_ID'
        self.household_id = 'Household_ID'
        self.report_id = 'Report_ID'
        self.timestamp = 'Timestamp'

        self.use_local_time = use_local_time
        self.warnings = not suppress_warning

        # check that all necessary files are available 
        self.__check__all_files_available__()

        # load meta data 
        self.global_meta_data = pd.read_csv(self.data_path+'meta_data/households.csv', sep=';')
        self.global_meta_data = pd.merge(self.global_meta_data, pd.read_csv(self.data_path+'meta_data/meta_data.csv', sep=';'), on=self.household_id, how='outer')
        self.global_meta_data.sort_values(by=self.household_id, inplace=True)
        self.global_meta_data.reset_index(drop=True, inplace=True)
        self.global_meta_data[self.household_id] = self.global_meta_data[self.household_id].astype('Int64')
        self.description_meta_data = pd.read_csv(self.data_path+'meta_data/meta_data_variables.csv', sep=';')

        # load overview data frames for weather data 
        self.global_weather_id_mapping = self.global_meta_data[[self.household_id, self.weather_id]].copy().dropna()
        self.global_weather_availability = pd.read_csv(self.data_path+'weather_data/overview/weather_variables_availability.csv', sep=';')
        self.description_weather = pd.read_csv(self.data_path+'weather_data/overview/weather_variables.csv', sep=';')

        # load overview data frames for smart meter data 
        self.global_smd_overview = pd.read_csv(self.data_path+'smart_meter_data/overview/smart_meter_data_15min_overview.csv', sep=';')
        self.global_smd_overview = pd.merge(self.global_smd_overview, pd.read_csv(self.data_path+'smart_meter_data/overview/smart_meter_data_daily_overview.csv', sep=';'), on=self.household_id, how='outer')
        self.global_smd_overview = pd.merge(self.global_smd_overview, pd.read_csv(self.data_path+'smart_meter_data/overview/smart_meter_data_monthly_overview.csv', sep=';'), on=self.household_id, how='outer')
        self.global_smd_overview = pd.merge(self.global_smd_overview, pd.read_csv(self.data_path+'smart_meter_data/overview/smart_meter_data_cumulative_counters_overview.csv', sep=';'), on=self.household_id, how='outer')
        self.global_smd_overview[self.household_id] = self.global_smd_overview[self.household_id].astype('Int64')
        self.global_smd_overview.sort_values(by=[self.household_id], inplace=True)
        self.global_smd_overview.reset_index(drop=True, inplace=True)

        for col in ['SMD_15min_TimeAvailable_EarliestTimestamp', 'SMD_15min_TimeAvailable_LatestTimestamp', 'SMD_daily_TimeAvailable_EarliestTimestamp', 'SMD_daily_TimeAvailable_LatestTimestamp', 'SMD_daily_TimeAvailable_LatestTimestamp', 'SMD_monthly_TimeAvailable_LatestTimestamp']:
            self.global_smd_overview[col] = pd.to_datetime(self.global_smd_overview[col], utc=True)
            if self.use_local_time: 
                self.global_smd_overview = self.__convert_UTC_to_local_time__(self.global_smd_overview, inplace=True, timestamp_column=col)
        
        # load protocols
        self.description_protocols = pd.read_csv(self.data_path+'reports/protocols_variables.csv', sep=';')
        self.protocols = pd.read_csv(self.data_path+'reports/protocols.csv', sep=';')
        self.protocols['Visit_Date'] = pd.to_datetime(self.protocols['Visit_Date'], utc=True)
        if self.use_local_time: 
            self.protocols = self.__convert_UTC_to_local_time__(self.protocols, inplace=True, timestamp_column='Visit_Date')
        self.protocols['Visit_Date'] = self.protocols['Visit_Date'].dt.date
        self.protocols[self.report_id] = self.protocols[self.report_id].astype(str)
        self.protocols.loc[self.protocols[self.household_id].isna(), self.household_id] = pd.NA
        self.protocols[self.household_id] = self.protocols[self.household_id].astype('Int64')
    
    def __check__all_files_available__(self): 
        '''
            Checks for completeness of the data set, i.e. if all files under the given path are available. 
        '''

        # check for the right folder structure 
        folders = [ 
            'meta_data/', 
            'reports/', 
            'smart_meter_data/15min/', 
            'smart_meter_data/daily/', 
            'smart_meter_data/monthly/',
            'smart_meter_data/cumulative_counters/',
            'smart_meter_data/overview/',
            'weather_data/daily/', 
            'weather_data/hourly/',
            'weather_data/overview/'
        ]
        for folder in folders:
            assert os.path.isdir(self.data_path + folder), 'HEAPO.__check__all_files_available__(): Missing folder {}'.format(self.data_path+folder)
        
        # check that the right files are available
        files = [
            'meta_data/households.csv',
            'meta_data/meta_data.csv',
            'meta_data/meta_data_variables.csv',

            'reports/protocols_variables.csv', 
            'reports/protocols.csv',

            'smart_meter_data/overview/smart_meter_data_15min_overview.csv', 
            'smart_meter_data/overview/smart_meter_data_daily_overview.csv', 
            'smart_meter_data/overview/smart_meter_data_monthly_overview.csv', 
            'smart_meter_data/overview/smart_meter_data_cumulative_counters_overview.csv', 

            'weather_data/overview/weather_variables.csv',
            'weather_data/overview/weather_variables_availability.csv',

        ] 
        for file in files: 
            assert os.path.isfile(self.data_path + file), 'HEAPO.__check__all_files_available__(): Missing file {}'.format(self.data_path+file)

        # check that the weather data is complete
        weather_ids = pd.read_csv(self.data_path+'weather_data/overview/weather_variables_availability.csv', sep=';')[self.weather_id].unique()
        for wid in weather_ids:
            assert os.path.isfile(self.data_path + 'weather_data/daily/{}.csv'.format(wid)), 'HEAPO.__check__all_files_available__(): Missing file {}'.format(self.data_path+'weather_data/daily/'+str(wid)+'.csv')
            assert os.path.isfile(self.data_path + 'weather_data/hourly/{}.csv'.format(wid)), 'HEAPO.__check__all_files_available__(): Missing file {}'.format(self.data_path+'weather_data_hourly/'+str(wid)+'.csv')

        # check that all smart meter data files exist
        for resolution in ['15min', 'daily', 'monthly', 'cumulative_counters']:
            household_ids = pd.read_csv(self.data_path+'smart_meter_data/overview/smart_meter_data_{}_overview.csv'.format(resolution), sep=';')[self.household_id].unique()
            for hid in household_ids:
                assert os.path.isfile(self.data_path + 'smart_meter_data/{}/{}.csv'.format(resolution, hid)), 'HEAPO.__check__all_files_available__(): Missing file {}'.format(self.data_path+'smart_meter_data/15_min/'+str(hid)+'.csv')
    
    def __convert_id_to_float__(self, id):
        '''
            Converts a string identifier to a float identifier. 
            Args: 
                id: string identifier
            Returns: 
                float identifier
        '''
        try: 
            if not isinstance(id, float): 
                id = float(id)
        except: 
            pass # return None
        return id
    
    def __convert_id_to_int__(self, id):
        '''
            Converts a string identifier to an integer identifier. 
            Args: 
                id: string identifier
            Returns: 
                integer identifier
        '''
        try: 
            if not isinstance(id, int): 
                id = int(id)
        except: 
            pass # return None
        return id

    def __check_allowed_resolutions__(self, resolution:str, weather:bool=False):
        '''
            Checks if the resolution is allowed for the data set. 
            Args: 
                resolution: string identifier of resolution
                weather: boolean to indicate if the resolution is for weather data or smart meter data
        '''
        if weather: 
            res = ['daily', 'hourly', 'hourly_and_daily']
            assert resolution in res, 'HEAPO.__check_allowed_resolutions__(): Resolution {} is not supported for the weather data. Only the following is supported: {}'.format(resolution, res)
        else: 
            res = ['15min', 'daily', 'monthly', 'cumulative_counters']
            assert resolution in res, 'HEAPO.__check_allowed_resolutions__(): Resolution {} is not supported for the smart meter data. Only the following is supported: {}'.format(resolution, res)
    
    def __convert_UTC_to_local_time__(self, df, inplace:bool=False, timestamp_column=None):
        '''
            Helper function to convert UTC time to local time (Europe/Zurich). 
            NOTE: not intended to be used by user directly - too little assertions. 
            Args: 
                df: pd.DataFrame to be converted
                inplace: bool, if True, the conversion is done in place, otherwise a copy is returned
                timestamp_column: string identifier of the timestamp column
            Returns:
                df: pd.DataFrame
        '''

        if timestamp_column is None:
            timestamp_column = self.timestamp
        assert isinstance(df, pd.DataFrame), 'HEAPO.__convert_UTC_to_local_time__(): The input is not of type pandas DataFrame.'
        assert pd.api.types.is_datetime64_any_dtype(df[timestamp_column]), 'HEAPO.__convert_UTC_to_local_time__(): The timestamp column is not of type pandas datetime.'

        # make a copy if not handled in place
        if not inplace:
            df = df.copy()

        # if not yet timestamp aware, set it as UTC
        if df[timestamp_column].dt.tz is None:
            df[timestamp_column] = df[timestamp_column].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        # if already local Zurich time, return
        elif str(df[timestamp_column].dt.tz) == 'Europe/Zurich':
            return df
        # otherwise make sure that it is not any other local time but UTC
        else: 
            assert str(df[timestamp_column].dt.tz) == 'UTC', 'convert_UTC_to_local_time(): Timestamp is not UTC but {}'.format(df[timestamp_column].dt.tz)
        # convert to Zurich time
        df[timestamp_column] = df[timestamp_column].dt.tz_convert('Europe/Zurich')
        assert str(df[timestamp_column].dt.tz) == 'Europe/Zurich' # check that everything went correctly
        return df

    def __convert_local_time_to_UTC__(self, df, inplace:bool=False, timestamp_column=None):
        '''
            Helper function to convert local time (Europe/Zurich) to UTC. 
            NOTE: not intended to be used by user directly - too little assertions. 
            Args: 
                df: pd.DataFrame to be converted
                inplace: bool, if True, the conversion is done in place, otherwise a copy is returned
                timestamp_column: string identifier of the timestamp column
            Returns:
                df: pd.DataFrame
        '''

        if timestamp_column is None:
            timestamp_column = self.timestamp
        assert isinstance(df, pd.DataFrame), 'HEAPO.__convert_local_time_to_UTC__(): The input is not of type pandas DataFrame.'
        assert pd.api.types.is_datetime64_any_dtype(df[timestamp_column]), 'HEAPO.__convert_local_time_to_UTC__(): The timestamp column is not of type pandas datetime.'

        # make a copy if not handled in place
        if not inplace:
            df = df.copy()

        # if not yet timestamp aware, set it as UTC
        if df[timestamp_column].dt.tz is None:
            df[timestamp_column] = df[timestamp_column].dt.tz_localize('Europe/Zurich', ambiguous='NaT', nonexistent='NaT')
        # if already UTC, return
        elif str(df[timestamp_column].dt.tz) == 'UTC':
            return df
        # otherwise make sure that it is not any other local time but Europe/Zurich
        else: 
            assert str(df[timestamp_column].dt.tz) == 'Europe/Zurich', 'convert_local_time_to_UTC(): Timestamp is not UTC but {}'.format(df[timestamp_column].dt.tz)
        # convert to UTC
        df[timestamp_column] = df[timestamp_column].dt.tz_convert('UTC')
        assert str(df[timestamp_column].dt.tz) == 'UTC'
        return df

    def __merge_smart_meter_and_weather_data__(self, df_smd:pd.DataFrame, df_weather:pd.DataFrame, interpolate_hourly:bool=False):
        '''
            This is a helper function to merge smart meter data and weather data when using combined loading.
            NOTE: this function is not intended to be used directly by the user, therefore it may be missing assertions.
            Args: 
                df_smd: data frame of smart meter data
                df_weather: data frame of weather data
                interpolate_hourly: boolean to indicate if hourly weather data should be interpolated
                    NOTE: interpolation will only be done between two consecutive hours
            Returns: 
                data frame that contains the merged data
        '''
        assert self.timestamp in df_smd.columns.values, 'HEAPO.merge_smart_meter_and_weather_data(): Timestamp column not found in smart meter data.'
        assert self.timestamp in df_weather.columns.values, 'HEAPO.merge_smart_meter_and_weather_data(): Timestamp column not found in weather data.'
        # NOTE: no assertion that the timestamp is of type datetime, because this is already done in the loading functions

        # get sets of relevant columns 
        smd_original_columns = df_smd.columns.values.tolist() # original smart meter data columns 
        # NOTE: the following assumes that the weather variables contain in their name whether they are of hourly or daily resolution
        # NOTE: this is valid because it is assumed that the user does not change the column names of the weather data provided in the HEAPO data set
        weather_hourly_columns = [e for e in df_weather.columns.values if 'hourly' in e] # hourly weather data columns
        weather_daily_columns = [e for e in df_weather.columns.values if 'daily' in e]
        weather_columns = [e for e in df_weather.columns.values if e not in [self.timestamp, 'Date', self.weather_id]]
        assert len(weather_hourly_columns) > 0 or len(weather_daily_columns) > 0, 'HEAPO.merge_smart_meter_and_weather_data(): No weather data columns found in weather data.'
        assert all([e in weather_daily_columns or e in weather_hourly_columns for e in weather_columns]), 'HEAPO.merge_smart_meter_and_weather_data(): Not all weather columns are either daily or hourly.'
        
        # create a copy of the data frames to be used
        df_smd_copy = df_smd.copy()
        df_weather_copy = df_weather.copy()

        # NOTE: it is not checked whether the resolution of smart meter data is below or equal hourly 
        # NOTE: current implementation only works if there are not multiple smart meter data frames and weather data frames combined in one and then merged --> would require consideration of ID for merging

        # add date columns 
        df_smd_copy['Date'] = df_smd_copy[self.timestamp].dt.date
        df_weather_copy['Date'] = df_weather_copy[self.timestamp].dt.date

        # merge the hourly weather data on the hourly values and interpolate
        if len(weather_hourly_columns) > 0: 
            
            # in the case of interpolation, merge only on the exact hour and then interpolate between neighboring hours 
            # NOTE: limit parameter is configured to fill 3 consecutive values only - i.e., assuming data in 15-min resolution and filling only for the missing 45 min between two consecutive measurements per hour
            if interpolate_hourly: 
                df_smd_copy = pd.merge(df_smd_copy, df_weather_copy[weather_hourly_columns + [self.timestamp]].drop_duplicates(), on=self.timestamp, how='left')
                df_smd_copy.set_index(self.timestamp, inplace=True)
                df_smd_copy[weather_hourly_columns] = df_smd_copy[weather_hourly_columns].interpolate(method='time', limit_direction='both', limit_area='inside', limit=3)
                df_smd_copy.reset_index(inplace=True)

            # in the other case just add the exact value of each hour 
            else: 
                df_smd_copy['Hour'] = df_smd_copy[self.timestamp].dt.hour.astype(float)
                df_smd_copy['Hour'] =  df_smd_copy['Hour'].astype(float)

                df_weather_copy['Hour'] = df_weather_copy[self.timestamp].dt.hour
                df_weather_copy['Hour'] =  df_weather_copy['Hour'].astype(float)

                df_smd_copy = pd.merge(df_smd_copy, df_weather_copy[weather_hourly_columns + ['Date', 'Hour']].drop_duplicates(), on=['Date', 'Hour'], how='left')
                df_smd_copy.drop(columns=['Hour'], inplace=True)

        # merge the daily weather data on the daily values without interpolation
        if len(weather_daily_columns) > 0: 
            df_smd_copy = pd.merge(df_smd_copy, df_weather_copy[weather_daily_columns + ['Date']].drop_duplicates(), on='Date', how='left')
        
        # bring columns back into the right order 
        df_smd_copy.drop(columns=['Date'], inplace=True)
        df_smd_copy = df_smd_copy[smd_original_columns + weather_columns]
        return df_smd_copy
    
    def __encode_time_as_2D_cyclic_feature__(self, time, scale):
        '''
            Maps time onto 2D plane, i.e. encodes it as cyclic features to better reflect periodicity.
            Args: 
                time (int): time or date to be encoded
                scale (int): for time of day: 24*60, for day of year: 365, for day of week: 7
            Returns:
                numpy array with two columns, first colum = x values and second column = y values
        '''
        return np.sin(2 * np.pi*time/scale), np.cos(2 * np.pi*time/scale)
    
    def __public_holidays_canton_zurich__(self): 
        '''
            Returns a list of public holidays in the canton of Zurich for the years 2016-2020.
            NOTE: some regions may have additional public holidays, which are not covered here.
        '''
        return pd.to_datetime(pd.Series([
            '2019-01-01', # Neujahrstag
            '2019-04-19', # Karfreitag
            '2019-04-22', # Ostermontag
            '2019-05-01', # Tag der Arbeit
            '2019-05-30', # Auffahrt
            '2019-06-10', # Pfingstmontag
            '2019-08-01', # Schweizer Nationalfeiertag
            '2019-12-25', # Weihnachten
            '2019-12-26', # Stephanstag

            '2020-01-01', # Neujahrstag
            '2020-04-10', # Karfreitag
            '2020-04-13', # Ostermontag
            '2020-05-01', # Tag der Arbeit
            '2020-05-21', # Auffahrt
            '2020-06-01', # Pfingstmontag
            '2020-08-01', # Schweizer Nationalfeiertag
            '2020-12-25', # Weihnachten
            '2020-12-26', # Stephanstag

            '2021-01-01', # Neujahrstag
            '2021-04-02', # Karfreitag
            '2021-04-05', # Ostermontag
            '2021-05-01', # Tag der Arbeit
            '2021-05-13', # Auffahrt
            '2021-05-24', # Pfingstmontag
            '2021-08-01', # Schweizer Nationalfeiertag
            '2021-12-25', # Weihnachten
            '2021-12-26', # Stephanstag

            '2022-01-01', # Neujahrstag
            '2022-04-15', # Karfreitag
            '2022-04-18', # Ostermontag
            '2022-05-01', # Tag der Arbeit
            '2022-05-26', # Auffahrt
            '2022-06-06', # Pfingstmontag
            '2022-08-01', # Schweizer Nationalfeiertag
            '2022-12-25', # Weihnachten
            '2022-12-26', # Stephanstag

            '2023-01-01', # Neujahrstag
            '2023-04-07', # Karfreitag
            '2023-04-10', # Ostermontag
            '2023-05-01', # Tag der Arbeit
            '2023-05-18', # Auffahrt
            '2023-05-29', # Pfingstmontag
            '2023-08-01', # Schweizer Nationalfeiertag
            '2023-12-25', # Weihnachten
            '2023-12-26', # Stephanstag

            '2024-01-01', # Neujahrstag
        ]
        )).dt.date.values.tolist()
    
    def __infer_temporal_resolution_in_minutes__(self, df:pd.DataFrame, timestamp_column:str=None):
        '''
            Helper function to infer the temporal resolution of a dataframe in minutes. 
            Args: 
                df: dataframe to infer the resolution from
                timestamp_column: column name with the timestamp - must be of type datetime - if None, HEAPO.timestamp is used
            Returns:
                resolution: (float) the temporal resolution in minutes
        '''
        if timestamp_column is None:
            timestamp_column = self.timestamp

        assert timestamp_column in df.columns, 'HEAPO.__infer_temporal_resolution_in_minutes__(): Column {} not in dataframe'.format(timestamp_column)
        assert pd.api.types.is_datetime64_any_dtype(df[timestamp_column]), 'HEAPO.__infer_temporal_resolution_in_minutes__(): Cannot infer resolution. The timestamp column is not of type pandas datetime.'

        # infer resolution 
        resolution = df.sort_values(timestamp_column)[timestamp_column].diff().min().total_seconds()/60.0
        return resolution

    def __create_heatmap_data__(self, df:pd.DataFrame, value_column:str, timestamp_column:str=None, add_missing_times:bool=True):
        '''
            Helper function for plotting heat maps. 
            Creates a heatmap data from a dataframe with a datetime index - i.e. creates a 2D representation of the data.
            Args: 
                df: dataframe containing the data to plot
                value_column: column name with values to create the heat map data for - must be numeric 
                timestamp_column: column name with the timestamp - must be of type datetime - if None, HEAPO.timestamp is used
                add_missing_times: (bool) if True, missing timestamps are added to the dataframe and filled with NAN - done by resampling 
            Returns:
                data: (dataframe) dataframe with time as index and date as columns, i.e. 2D representation of the data
        '''
        if timestamp_column is None:
            timestamp_column = self.timestamp

        assert value_column in df.columns, 'HEAPO.__create_heatmap_data__(): Column {} not in dataframe'.format(value_column)
        assert timestamp_column in df.columns, 'HEAPO.__create_heatmap_data__(): Column {} not in dataframe'.format(timestamp_column)
        assert pd.api.types.is_datetime64_any_dtype(df[timestamp_column]), 'HEAPO.__create_heatmap_data__(): Cannot create heatmap data. The timestamp column is not of type pandas datetime.'
        assert pd.api.types.is_numeric_dtype(df[value_column]), 'HEAPO.__create_heatmap_data__(): Cannot create heatmap data. The column to create the heat map from ({}) is not of type numeric.'.format(value_column)
        assert isinstance(add_missing_times, bool), 'HEAPO.__create_heatmap_data__(): The argument add_missing_days must be of type bool.'

        # create copy 
        data_df = df[[timestamp_column, value_column]].copy()

        # sort values 
        data_df = data_df.sort_values(timestamp_column)

        # automatically infer the resolution of the current dataframe in minutes
        time_delta_minutes = self.__infer_temporal_resolution_in_minutes__(df, timestamp_column) # data_df[timestamp_column].diff().min().total_seconds()/60.0
        assert time_delta_minutes <= 60.0*23.0, 'HEAPO.__create_heatmap_data__(): Cannot create heamap data because the current resolution of the data is greater than one day. Current resolution in minutes: {}'.format(time_delta_minutes)
        assert time_delta_minutes >= 1.0, 'HEAPO.__create_heatmap_data__(): The current resolution of the data is smaller than 1 minute. Please check the data for duplicates and remove these if existent.'

        # do resampling to add missing values
        if add_missing_times: 
            # convert to UTC first 
            data_df = self.__convert_local_time_to_UTC__(data_df, inplace=True, timestamp_column=timestamp_column)

            # resample data to fill missing days
            data_df = data_df.set_index(timestamp_column).resample('{}min'.format(time_delta_minutes)).first().reset_index().reindex(columns=df.columns)

            # previously missing days should be set to NAN
            data_df.loc[~data_df[timestamp_column].isin(df[timestamp_column].unique()), value_column] = np.nan

            # convert back to local time
            if self.use_local_time:
                data_df = self.__convert_UTC_to_local_time__(data_df, inplace=True, timestamp_column=timestamp_column)

        # pivot dates and times to create a two dimensional representation
        data_df["date"] = data_df[timestamp_column].dt.date
        data_df["time"] = data_df[timestamp_column].dt.time

        data_df = data_df.pivot_table(index='time', columns='date', values=value_column, dropna=False)  
        return data_df

    def get_keyword_weather_id(self): 
        '''
            Returns keyword to call the weather ID of this data set.
        '''
        return self.weather_id
    
    def get_keyword_household_id(self): 
        '''
            Returns keyword to call the household ID column of this data set.
        '''
        return self.household_id

    def get_keyword_timestamp(self): 
        '''
            Returns keyword to call the timestamp column of this data set.
        '''
        return self.timestamp
    
    def get_keyword_report_id(self): 
        '''
            Returns keyword to call the report_id column of this data set.
        '''
        return self.report_id
    
    def get_all_households(self, group:str=None): 
        '''
            Args: 
                group: string identifier of group, either 'treatment' or 'control' or None to select all 
            Returns list of all available households.
        '''
        if group is not None: 
            assert group in ['treatment', 'control'], 'HEAPO.get_all_households(): Group must be either "treatment" or "control".'
        assert isinstance(self.global_meta_data, pd.DataFrame) and self.household_id in self.global_meta_data.columns.values, 'HEAPO.get_all_households(): Smart meter data overview is not available or does not contain the household ID column.'
        if group is None: 
            return self.global_meta_data[self.global_meta_data[self.household_id].notna()][self.household_id].unique().tolist()
        elif group == 'treatment': 
            return self.global_meta_data[(self.global_meta_data[self.household_id].notna()) & (self.global_meta_data['Group']=='treatment')][self.household_id].unique().tolist()
        else: 
            return self.global_meta_data[(self.global_meta_data[self.household_id].notna()) & (self.global_meta_data['Group']=='control')][self.household_id].unique().tolist()
    
    def get_all_households_with_protocols(self):
        '''
            Returns list of all households for which protocols are available.
        '''
        assert isinstance(self.global_meta_data, pd.DataFrame) and self.household_id in self.global_meta_data.columns.values, 'HEAPO.get_all_households_with_protocols(): Meta data is not available or does not contain the household ID column.'
        return self.global_meta_data[self.global_meta_data['Protocols_Available']==True][self.household_id].unique().tolist()
    
    def get_all_households_with_multiple_protocols(self):
        '''
            Returns list of all households for which multiple protocols are available.
        '''
        assert isinstance(self.global_meta_data, pd.DataFrame) and self.household_id in self.global_meta_data.columns.values, 'HEAPO.get_all_households_with_multiple_protocols(): Meta data is not available or does not contain the household ID column.'
        return self.global_meta_data[self.global_meta_data['Protocols_HasMultipleVisits']==True][self.household_id].unique().tolist()

    def get_all_weather_ids(self): 
        '''
            Returns list of all available weather IDs.
        '''
        assert isinstance(self.global_weather_availability, pd.DataFrame) and self.weather_id in self.global_weather_availability.columns.values, 'HEAPO.get_all_weather_ids(): Weather data overview is not available or does not contain the weather ID column.'
        return self.global_weather_availability[self.weather_id].unique().tolist()
    
    def get_all_protocol_ids(self):
        '''
            Returns list of all available report IDs.
        '''
        assert isinstance(self.protocols, pd.DataFrame) and self.report_id in self.protocols.columns.values, 'HEAPO.get_all_protocol_ids(): Protocols are not available or do not contain the report ID column.'
        return self.protocols[self.report_id].unique().tolist()

    def household_exists(self, household_id):
        '''
            Checks if a household exists in the data set.
            Args: 
                household_id: string identifier of household
            Returns: 
                True if household exists, else False
        '''
        return self.__convert_id_to_float__(household_id) in self.get_all_households()
    
    def report_exists(self, report_id): 
        '''
            Checks if report ID exists in the data set.
            Args: 
                report_id: string identifier of report
            Returns: 
                True if report exists, else False
        '''
        assert isinstance(self.protocols, pd.DataFrame) and self.report_id in self.protocols.columns.values, 'HEAPO.report_exists(): Protocols are not available or do not contain the household ID column.'
        return str(report_id) in [str(e) for e in self.protocols[self.report_id].values]
    
    def weather_data_exists(self, weather_id):
        '''
            Checks if weather data file exists for specific weather ID.
            Args: 
                weather_id: string identifier of weather ID
            Returns: 
                True if weather data exists for weather ID, else False
        '''
        assert isinstance(self.global_weather_availability, pd.DataFrame) and self.weather_id in self.global_weather_availability.columns.values, 'HEAPO.weather_data_exists(): Weather data overview is not available or does not contain the weather ID column.'
        return str(weather_id) in [str(e) for e in self.global_weather_availability[self.weather_id].values]

    def load_smart_meter_data(self, household_id, resolution='15min'): 
        '''
            Loads smart meter data of a single household or multiple households in desired resolution if it exists.
            Args: 
                household_id: identifier of household
                resolution: string identifier of resolution
                    - '15min': 15 minute smart meter data
                    - 'daily': daily smart meter data
                    - 'monthly': monthly smart meter data
                    - 'cumulative_counters': cumulative daily counters of smart meters
            Returns: 
                data frame that contains the smart_meta data of the corresponding household if it exists, else None
        '''

        # check if resolution is valid 
        self.__check_allowed_resolutions__(resolution, weather=False)

        # check if household exists
        if self.household_exists(household_id):
            # check if file exists and load 
            pth = self.data_path + 'smart_meter_data/{}/{}.csv'.format(resolution, self.__convert_id_to_int__(household_id)) 
            assert os.path.isfile(pth), 'HEAPO.load_smart_meter_data(): File for given household ({}) cannot be found: {}'.format(household_id, pth)
            df = pd.read_csv(pth, sep=';', parse_dates=['Timestamp'])
            if self.use_local_time: 
                df = self.__convert_UTC_to_local_time__(df, inplace=True)
            return df
        else: 
            if self.warning: 
                warnings.warn('HEAPO.load_smart_meter_data(): Returning None for given household ID: {}'.format(household_id))
            return None 
        
    def load_smart_meter_data_multiple(self, household_ids, resolution='15min'):
        '''
            Loads smart meter data of multiple households in desired resolution if it exists and concatenates.
            NOTE: may be a bit slow for a very large number of households.
            NOTE: removes duplicates and ignores observations where file cannot be found.
            Args: 
                household_ids: list of identifiers of households
                resolution: string identifier of resolution
                    - '15min': 15 minute smart meter data
                    - 'daily': daily smart meter data
                    - 'monthly': monthly smart meter data
                    - 'cumulative_counters': cumulative daily counters of smart meters

            Returns: 
                data frame that contains the smart_meta data of the corresponding households if it exists, else None
        '''
        # check if resolution is valid 
        self.__check_allowed_resolutions__(resolution, weather=False)

        df_final = pd.DataFrame()
        for hid in household_ids:
            df = self.load_smart_meter_data(hid, resolution)
            if df is not None: 
                df_final = pd.concat([df_final, df], axis=0)
        df_final.drop_duplicates(inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        if self.use_local_time: 
            df_final = self.__convert_UTC_to_local_time__(df_final, inplace=True)
        return df_final
    
    def load_weather_data(self, id, resolution='daily'): 
        '''
            Loads weather data of household_id or weather_id if it exists.
            Args: 
                id: string identifier of weather_id or household_id
                resolution: string identifier of resolution
                    - 'daily': daily weather data
                    - 'hourly': hourly weather data
                    - 'hourly_and_daily': hourly weather data with daily data in one data frame
            Returns: 
                data frame that contains the weather data of the corresponding metering code if it exists, else None
        '''
        # check if resolution is valid
        self.__check_allowed_resolutions__(resolution, weather=True)

        # first check if id is a weather_id or household_id and in the second case try to find the right household_id
        if not self.weather_data_exists(id): 
            assert self.household_exists(id), 'HEAPO.load_weather_data(): Given ID [{}] is neither a weather ID nor a household ID.'.format(id)
            if self.__convert_id_to_float__(id) in self.global_weather_id_mapping[self.household_id].values:
                id = self.global_weather_id_mapping[self.global_weather_id_mapping[self.household_id] == self.__convert_id_to_float__(id)][self.weather_id].values[0]
            else: 
                if self.warning: 
                    warnings.warn('HEAPO.load_weather_data(): Returning None for ID: {}'.format(id))
                return None 
            
        # check if file exists and load
        if resolution != 'hourly_and_daily': # if just daily or hourly data, load directly
            assert os.path.isfile(self.data_path + 'weather_data/{}/{}.csv'.format(resolution, id)), 'HEAPO.load_weather_data(): File for given weather ID ({}) cannot be found: {}'.format(id, self.data_path + 'weather_data/{}/{}.csv'.format(resolution, id))
            return pd.read_csv(self.data_path + 'weather_data/{}/{}.csv'.format(resolution, id), sep=';', parse_dates=[self.timestamp])
        else: # otherwise load both and fuse them
            assert os.path.isfile(self.data_path + 'weather_data/daily/{}.csv'.format(id)), 'HEAPO.load_weather_data(): File for given weather ID ({}) cannot be found: {}'.format(id, self.data_path + 'weather_data/daily/{}.csv'.format(id))
            assert os.path.isfile(self.data_path + 'weather_data/hourly/{}.csv'.format(id)), 'HEAPO.load_weather_data(): File for given weather ID ({}) cannot be found: {}'.format(id, self.data_path + 'weather_data/hourly/{}.csv'.format(id))
            df_daily = pd.read_csv(self.data_path + 'weather_data/daily/{}.csv'.format(id), sep=';', parse_dates=[self.timestamp]).rename(columns={self.timestamp: 'Date'})
            df_daily['Date'] = df_daily['Date'].dt.date
            df_hourly = pd.read_csv(self.data_path + 'weather_data/hourly/{}.csv'.format(id), sep=';', parse_dates=[self.timestamp])
            df_hourly['Date'] = df_hourly[self.timestamp].dt.date
            df_hourly = pd.merge(df_hourly, df_daily.drop(columns=[self.weather_id]), on='Date', how='left')
            df_hourly = df_hourly[[self.weather_id, self.timestamp] + self.global_weather_availability.columns.values.tolist()[1:]] # bring data into the right order 
            if self.use_local_time: 
                df_hourly = self.__convert_UTC_to_local_time__(df_hourly, inplace=True)
            return df_hourly
        
    def load_weather_data_multiple(self, ids, resolution='daily'):
        '''
            Loads weather data of multiple households in desired resolution if it exists and concatenates.
            NOTE: may be a bit slow for a very large number of households.
            NOTE: removes duplicates and ignores observations where file cannot be found.
            Args: 
                ids: list of identifiers of households
                resolution: string identifier of resolution
                    - 'daily': daily weather data
                    - 'hourly': hourly weather data
                    - 'hourly_and_daily': hourly weather data with daily data in one data frame
            Returns: 
                data frame that contains the weather data of the corresponding households if it exists, else None
        '''
        # check if resolution is valid
        self.__check_allowed_resolutions__(resolution, weather=True)

        df_final = pd.DataFrame()
        for wid in ids:
            df = self.load_weather_data(wid, resolution)
            if df is not None: 
                df_final = pd.concat([df_final, df], axis=0)
        df_final.drop_duplicates(inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        # if self.use_local_time: 
        #     df_final = self.__convert_UTC_to_local_time__(df_final, inplace=True)
        return df_final

    def load_protocol_data(self, id):
        '''
            Loads protocol data for a given report id if available. 
            Args: 
                id: string identifier of report id or household id
            Returns: 
                data frame that contains the report data of the corresponding report id if it exists, else None
        '''

        # first try to find the report id in the protocols
        if self.report_exists(id): 
            df = self.protocols[self.protocols[self.report_id] == str(id)]
        else: # try to find household_id and then the report id
            assert self.household_exists(id), 'HEAPO.load_report_data(): Given ID [{}] is neither a report ID nor a household ID.'.format(id)
            df = self.protocols[self.protocols[self.household_id] == self.__convert_id_to_float__(id)]
            if len(df) == 0:
                df = None
        if df is None and self.warning:
            warnings.warn('HEAPO.load_report_data(): Returning None for ID: {}'.format(id))
        return df 
    
    def load_protocol_data_multiple(self, ids, add_comments:bool=False):
        '''
            Loads protocol data for multiple report ids if available and concatenates. 
            NOTE: may be a bit slow for a very large number of households.
            NOTE: removes duplicates and ignores observations where file cannot be found.
            Args: 
                ids: list of report ids or household ids
                comments: boolean to indicate if comments should be loaded as well and added as separate columns
            Returns: 
                data frame that contains the report data of the corresponding report ids if it exists, else None
        '''
        df_final = pd.DataFrame()
        for rid in ids:
            df = self.load_protocol_data(rid, add_comments)
            if df is not None: 
                df_final = pd.concat([df_final, df], axis=0)
        df_final.drop_duplicates(inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        return df_final
    
    def load_smart_meter_and_weather_data_combined(self, household_id, smd_resolution:str='15min', weather_resolution:str='hourly_and_daily', interpolate_hourly:bool=True): 
        '''
            Combined data loading of smart meter data and weather data for a single household.
            Args: 
                household_id: identifier of household
                smd_resolution: string identifier of resolution for smart meter data
                    - '15min': 15 minute smart meter data
                    - 'daily': daily smart meter data
                    - 'cumulative_counters': cumulative daily counters of smart meters
                weather_resolution: string identifier of resolution for weather data
                    - 'daily': daily weather data
                    - 'hourly': hourly weather data
                    - 'hourly_and_daily': hourly weather data with daily data in one data frame
                interpolate_hourly: boolean to indicate if hourly weather data should be interpolated
                    NOTE: interpolation will only be done between for the 15-minute intervals between two consecutive hours
            Returns:
                data frame that contains the merged data if both exist, smart meter data and weather data, else None
        '''
        
        # check for allowed resolution and their combinations 
        self.__check_allowed_resolutions__(smd_resolution, weather=False)
        self.__check_allowed_resolutions__(weather_resolution, weather=True)
        assert smd_resolution != 'monthly', 'HEAPO.load_smart_meter_and_weather_data_combined(): Monthly resolution for smart meter data is not supported for combined loading with weather. Please load smart meter data and weather data separately and handle combination yourself.'
        if smd_resolution == 'daily' or smd_resolution == 'cumulative_counters': 
            assert weather_resolution == 'daily', 'HEAPO.load_smart_meter_and_weather_data_combined(): If the smart meter data should be loaded in daily resolution, the weather data resolution must also be configured to be daily.'
        
        # load smart meter data and weather data in UTC - in dependent of the local time setting
        original_local_time_setting = self.use_local_time
        self.use_local_time = False

        # load smart meter data and weather data
        df_smd = self.load_smart_meter_data(household_id, resolution=smd_resolution)
        df_weather = self.load_weather_data(household_id, resolution=weather_resolution)

        # reset local time setting
        self.use_local_time = original_local_time_setting

        if isinstance(df_smd, pd.DataFrame) and isinstance(df_weather, pd.DataFrame): 
            df = self.__merge_smart_meter_and_weather_data__(df_smd, df_weather, interpolate_hourly=interpolate_hourly)
            
            if self.use_local_time: # just now convert back to local time if this is desired
                df = self.__convert_UTC_to_local_time__(df, inplace=True)
            return df
        if self.warning: 
            warnings.warn('HEAPO.load_smart_meter_and_weather_data_combined(): Returning None for household ID: {}'.format(household_id))
        return None
    
    def load_smart_meter_and_weather_data_combined_multiple(self, household_ids, smd_resolution:str='15min', weather_resolution:str='hourly_and_daily', interpolate_hourly:bool=True):
        '''
            Combined data loading of smart meter data and weather data for multiple households. 
            NOTE: may be a bit slow for a very large number of households.
            NOTE: removes duplicates and ignores observations where file cannot be found.
            Args: 
                household_ids: list of household ids
                smd_resolution: string identifier of resolution for smart meter data
                    - '15min': 15 minute smart meter data
                    - 'daily': daily smart meter data
                weather_resolution: string identifier of resolution for weather data
                    - 'daily': daily weather data
                    - 'hourly': hourly weather data
                    - 'hourly_and_daily': hourly weather data with daily data in one data frame
                interpolate_hourly: boolean to indicate if hourly weather data should be interpolated
                    NOTE: interpolation will only be done between for the 15-minute intervals between two consecutive hours
            Returns: 
                data frame that contains the report data of the corresponding report ids if it exists, else None
        '''
        # assertion that household_ids is iterable 
        assert isinstance(household_ids, list) or isinstance(household_ids, np.ndarray), 'HEAPO.load_smart_meter_and_weather_data_combined_mutliple(): Parameter household_ids must be a list or numpy array.'
        df_final = pd.DataFrame()
        for id in household_ids:
            df = self.load_smart_meter_and_weather_data_combined(id, smd_resolution=smd_resolution, weather_resolution=weather_resolution, interpolate_hourly=interpolate_hourly)
            if df is not None: 
                df_final = pd.concat([df_final, df], axis=0)
        df_final.drop_duplicates(inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        return df_final
    
    def load_meta_data(self, household_id, return_dict:bool=True):
        '''
            Loads meta data for a given household id if available. 
            Args: 
                household_id: string identifier of household id
                return_dict: boolean to indicate if a dictionary should be returned - if False a data frame will be returned
            Returns: 
                if return_dict is False:
                    data frame that contains the meta data of the corresponding household if it exists, else None
                else: 
                    dictionary that contains the meta data of the corresponding household if it exists, else None
        '''
        assert isinstance(self.global_meta_data, pd.DataFrame), 'HEAPO.load_meta_data(): Meta data is not available.'
        if self.household_exists(household_id): 
            df_meta = self.global_meta_data[self.global_meta_data[self.household_id] == self.__convert_id_to_float__(household_id)].copy()
            if return_dict: 
                return df_meta.iloc[0].to_dict()
            else: 
                return df_meta
        if self.warnings: 
            warnings.warn('HEAPO.load_meta_data(): Returning None for household ID: {}'.format(household_id))
        return None

    def get_all_protocols(self): 
        '''
            Returns data frame that contains all protocols. None if not available.
        '''
        assert isinstance(self.protocols, pd.DataFrame), 'HEAPO.get_all_protocols(): Protocols are not available.'
        return self.protocols.copy()
    
    def get_protocols_description(self):
        '''
            Returns data frame that contains a description of all protocol variables. None if not available.
        '''
        assert isinstance(self.description_protocols, pd.DataFrame), 'HEAPO.get_protocols_description(): Protocol description is not available.'
        return self.description_protocols.copy()
    
    def get_weather_description(self):
        '''
            Returns data frame that contains a description of all weather variables. None if not available.
        '''
        assert isinstance(self.description_weather, pd.DataFrame), 'HEAPO.get_weather_description(): Weather description is not available.'
        return self.description_weather.copy()
    
    def get_weather_data_availability_overview(self):
        '''
            Returns data frame that contains an overview of the availability of weather data. None if not available.
        '''
        return self.global_weather_availability.copy()
    
    def get_available_weather_variables(self, id):
        '''
            Args:
                id: string identifier of weather_id or household_id
            Returns: 
                list of available weather variables for a given weather ID. None if not available.
        '''
        # first check if id is a weather_id or household_id and in the second case try to find the right household_id
        if not self.weather_data_exists(id): 
            assert self.household_exists(id), 'HEAPO.get_available_weather_variables(): Given ID [{}] is neither a weather ID nor a household ID.'.format(id)
            if self.__convert_id_to_float__(id) in self.global_weather_id_mapping[self.household_id].values:
                id = self.global_weather_id_mapping[self.global_weather_id_mapping[self.household_id] == self.__convert_id_to_float__(id)][self.weather_id].values[0]
            else: 
                if self.warnings: 
                    warnings.warn('HEAPO.get_available_weather_variables(): Returning None for ID: {}'.format(id))
                return None 
    
        # when the right weather ID has been found - find available variables 
        df_temp = self.global_weather_availability.T.reset_index().copy()
        df_temp.columns = df_temp.iloc[0]
        df_temp.drop(0, inplace=True)
        return df_temp[df_temp[id] == True][self.weather_id].values.tolist()
    
    def get_smart_meter_data_overview(self):
        '''
            Returns data frame that contains an overview of the availability of smart meter data. None if not available.
        '''
        assert isinstance(self.global_smd_overview, pd.DataFrame), 'HEAPO.get_smart_meter_data_overview(): Smart meter data overview is not available.'
        return self.global_smd_overview.copy()
    
    def get_meta_data_overview(self): 
        '''
            Returns data frame that contains an overview of the meta data. None if not available.
        '''
        assert isinstance(self.global_meta_data, pd.DataFrame), 'HEAPO.get_meta_data_overview(): Meta data overview is not available.'
        return self.global_meta_data.copy()
    
    def get_weather_id_mapping(self):
        '''
            Returns data frame that contains the mapping of household IDs to weather IDs. None if not available.
        '''
        return self.global_weather_id_mapping.copy()

    def explain_variable(self, variable:str, return_dict=True):
        '''
            Provides a description of a variable either relating to weather or protocols / comments. 
            Args:
                variable: string identifier of variable name
                return_dict: boolean to indicate if a dictionary should be returned - if False a data frame will be returned
            Returns:
                dictionary (or data frame) that contains information about the variable if it exists, else None
        '''
        if not isinstance(variable, str):
            return None

        # search through the description of the protocols 
        if variable in self.description_protocols['VariableName'].values:
            if return_dict: 
                return self.description_protocols.loc[self.description_protocols['VariableName'] == variable].iloc[0].to_dict()
            else: 
                return self.description_protocols[self.description_protocols['VariableName'] == variable]
        
        # search through the description of the weather data
        elif variable in self.description_weather['VariableName'].values:
            if return_dict: 
                return self.description_weather.loc[self.description_weather['VariableName'] == variable].iloc[0].to_dict()
            else: 
                return self.description_weather[self.description_weather['VariableName'] == variable]
            
        elif variable in self.description_meta_data['VariableName'].values:
            if return_dict: 
                return self.description_meta_data.loc[self.description_meta_data['VariableName'] == variable].iloc[0].to_dict()
            else: 
                return self.description_meta_data[self.description_meta_data['VariableName'] == variable]
            
        # return None if nothing was found
        return None 
    
    def add_temporal_information(self, df, inplace:bool=True, timestamp_column=None): 
        '''
            Adds temporal features of a data frame with timestamps for better filtering, e.g., this allows you to filter on desired seasons or times of the day. 
            Args: 
                df: data frame to be processed - needs to contain a timestamp column
                inplace: boolean to indicate if the processing should be done in place or a copy should be returned
                timestamp_column: string identifier of the timestamp column - if None: the timestamp identifier of the class will be used 
            Returns: 
                processed copy of the data frame with additional columns
        '''
        if timestamp_column is None: 
            timestamp_column = self.timestamp
        assert isinstance(df, pd.DataFrame), 'Cannot preprocess given data frame. Parameter df must be of type pd.DataFrame, but given was: {}'.format(type(df))
        assert timestamp_column in df.columns.values, 'Cannot preprocess given data frame. The timestamp column {} is not available in the given data frame'.format(timestamp_column)
        assert pd.api.types.is_datetime64_any_dtype(df[timestamp_column]), 'Cannot preprocess given data frame. The timestamp column is not of type pandas datetime.'

        # warn if the timestamp is not in local time
        # if (str(df[timestamp_column].dt.tz) != 'Europe/Zurich') and self.warnings:
        #     warnings.warn('HEAPO.add_temporal_information(): The timestamp column is not in local time (Europe/Zurich). Therefore, some temporal features calculated may not be well representative.')

        # create a copy if not handled in place
        if not inplace:
            df = df.copy()

        # calculate additional columns
        try: 

            df['Date'] = df[timestamp_column].dt.date
            df['Time'] = df[timestamp_column].dt.time
            df['Year'] = df[timestamp_column].dt.year
            df['Month'] = df[timestamp_column].dt.month
            df['Week'] = df[timestamp_column].dt.isocalendar().week
            df['DayOfWeek'] = df[timestamp_column].dt.dayofweek
            df['DayOfYear'] = df[timestamp_column].dt.dayofyear
            df['Hour'] = df[timestamp_column].dt.hour
            df['Minute'] = df[timestamp_column].dt.minute
            df['MinuteOfDay'] = df['Hour']*60 + df['Minute']

            # mapping if times of the day are met
            df['Weekday'] = np.where(df['DayOfWeek'] <=4, True, False)
            df['Weekend'] = np.where(df['DayOfWeek'] >=5, True, False)
            df['PublicHoliday'] = np.where(df['Date'].isin(self.__public_holidays_canton_zurich__()), True, False)
            df['Morning'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 10), True, False)
            df['Noon'] = np.where(((df['Hour'] >= 10) & (df['Hour'] < 14)), True, False)
            df['Afternoon'] = np.where((df['Hour'] >= 14) & (df['Hour'] < 18), True, False)
            df['Evening'] = np.where((df['Hour'] >= 18) & (df['Hour'] < 23), True, False)
            df['Day'] = np.where((df['Hour'] < 23) & (df['Hour'] >= 6), True, False)
            df['Night'] = np.where((df['Hour'] >= 23) | (df['Hour'] < 6), True, False)

            # mapping months to seasons: 1- winter, 2-spring, 3-summer, 4-autumn 
            seasons = {1: 1, 2: 1, 3: 2, 4: 2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1} # maps months to seasons: 1- winter, 2-spring, 3-summer, 4-autumn
            df['Season'] = df['Month'].map(seasons, na_action=None)
            df['Winter'] = np.where(df['Season'] == 1, True, False)
            df['Spring'] = np.where(df['Season'] == 2, True, False)
            df['Summer'] = np.where(df['Season'] == 3, True, False)
            df['Autumn'] = np.where(df['Season'] == 4, True, False)
            df['TransitionPeriod'] = np.where((df['Season'] == 2) | (df['Season'] == 4), True, False)

            # create cyclic features for time of day, day of year, and day of week, which reflect periodicity better
            x,y = self.__encode_time_as_2D_cyclic_feature__(df['MinuteOfDay'].values, 24*60)
            df['Cyclic_TimeOfDay_X'] = x
            df['Cyclic_TimeOfDay_Y'] = y
            x,y = self.__encode_time_as_2D_cyclic_feature__(df['DayOfYear'].values, 365)
            df['Cyclic_DayOfYear_X'] = x
            df['Cyclic_DayOfYear_Y'] = y
            x,y = self.__encode_time_as_2D_cyclic_feature__(df['DayOfWeek'].values, 7)
            df['Cyclic_DayOfWeek_X'] = x
            df['Cyclic_DayOfWeek_Y'] = y


        except Exception as e: 
            # when the columns cannot be calculated fill them as NAN
            if self.warnings: 
                warnings.warn('HEAPO.add_temporal_information(): Could not calculate relevant columns. Error: {}'.format(e))
            cols = [
                'Date', 
                'Time', 
                'Year', 
                'Month', 
                'Week', 
                'DayOfWeek', 
                'DayOfYear',
                'Hour', 
                'Minute', 
                'MinuteOfDay'
                'Weekday', 
                'Weekend',
                'PublicHoliday', 
                'Morning', 
                'Noon', 
                'Afternoon', 
                'Evening', 
                'Day', 
                'Night',
                'Season', 
                'Winter', 
                'Spring', 
                'Summer', 
                'Autumn', 
                'TransitionPeriod', 
                'Cyclic_TimeOfDay_X',
                'Cyclic_TimeOfDay_Y',
                'Cyclic_DayOfYear_X',
                'Cyclic_DayOfYear_Y',
                'Cyclic_DayOfWeek_X',
                'Cyclic_DayOfWeek_Y'
            ]
            for col in cols: 
                if col not in df.columns.values: 
                    df[col] = pd.NA

        return df
    
    def limit_time_series_around_date(self, df, report_date, months_before=None, months_after=None, inplace:bool=True):
        '''
            Limits the time range of a given data frame around a specific date. 
            NOTE: This may be useful when wanting to predict heat pump configurations. 
            NOTE: E.g., you may want to assume that configurations were the same as reported by energy consultant for X months before and X months after the report date.
            Args: 
                df: data frame that contains time series data 
                report_date: date around which the time series should be limited (either datetime.date object or string in format 'YYYY-MM-DD')
                months_before: number of months before the report date that should be included in the time series - will be ignored if None
                months_after: number of months after the report date that should be included in the time series - will be ignored if None 
                inplace: boolean to indicate if the processing should be done in place or a copy should be returned
            Returns: 
                data frame with limited time range if it worked, else None
        '''
        try:
            if not isinstance(report_date, datetime.date):
                report_date = pd.to_datetime(str(report_date), format='%Y-%m-%d').date()
            if not inplace:
                df = df.copy()
            if months_before is not None: 
                df = df[df[self.timestamp].dt.date >= report_date - relativedelta(months=months_before)] 
            if months_after is not None: 
                df = df[df[self.timestamp].dt.date <= report_date + relativedelta(months=months_after)] 
            if self.warnings and len(df) == 0:
                warnings.warn('HEAPO.limit_time_series_around_date(): The time series is empty after limiting it around the date {}. Returning empty data frame.'.format(report_date))
        except: 
            if self.warnings: 
                warnings.warn('HEAPO.limit_time_series_around_date(): Could not handle datetime limitation. Returning None.')
        return df
    
    def split_time_series_around_consultation(self, df:pd.DataFrame):
        '''
            Splits a time series into three parts: before, during, and after consultation.
            NOTE: assumes that the data frame contains a column named "AffectsTimePoint" and that it contains values "before visit" and "after visit"
            NOTE: ignores household IDs, i.e. if data frame contains multiple households, the split will be done for all of them
            Args: 
                df: data frame that contains time series data to be split 
            Returns: 
                sequence of three data frames: before, during, and after the consultation
        '''
        assert 'AffectsTimePoint' in df.columns.values, 'HEAPO.split_time_series_around_consultation(): The data frame does not contain a column named "AffectsTimePoint".'
        df_before = df[df['AffectsTimePoint'] == 'before visit'].copy()
        df_after = df[df['AffectsTimePoint'] == 'after visit'].copy()
        df_during = df[df['AffectsTimePoint'] == 'during visit'].copy()
        if self.warnings and len(df_before) == 0 and len(df_after) == 0 and len(df_during) == 0:
            warnings.warn('HEAPO.split_time_series_around_consultation(): Returning only empty data frames.')
        return df_before, df_during, df_after
    
    def change_resolution(self, df:pd.DataFrame, resolution:str, timestamp_column=None, add_missing_dates:bool=False, inplace:bool=True, skipna:bool=False): 
        ''' 
            Changes the resolution of a given data frame.
            NOTE: This may be useful for quality check of the data (e.g., load 15 min smart meter data and resample to daily data - then compare to original daily data)
            NOTE: generally may be a bit slow for a very large number of households.
            NOTE: can deal with combined data frames (e.g., smart meter and weather data), but currently not implemented to handle multiple households 
            NOTE: also if the function add_temporal_information() was used beforehand - recalculate it again own you own - as the time information will be wrong after resampling
            Args: 
                df: data frame that contains time series data to be resampled
                resolution: string identifier of resolution
                    - '15min': 15 minute smart meter data
                    - 'daily': daily smart meter data
                    - 'monthly': monthly smart meter data
                timestamp_column: string identifier of the timestamp column - if None: the timestamp identifier of the class will be used
                add_missing_dates: boolean to indicate if missing dates should be added to the data frame - otherwise they will be removed after resampling
                inplace: boolean to indicate if the processing should be done in place or a copy should be returned
                skipna: exclude NA/null values when computing the result
            Returns:
                resampled data frame 
        '''
        if timestamp_column is None: 
            timestamp_column = self.timestamp
        assert isinstance(df, pd.DataFrame), 'HEAPO.change_resolution(): Parameter df must be of type pd.DataFrame, but given was: {}'.format(type(df))
        assert timestamp_column in df.columns.values, 'HEAPO.change_resolution(): The timestamp column {} is not available in the given data frame'.format(timestamp_column)
        assert pd.api.types.is_datetime64_any_dtype(df[timestamp_column]), 'HEAPO.change_resolution(): The timestamp column is not of type pandas datetime.'

        # create a copy if not handled in place
        if not inplace:
            df = df.copy()
        
        # transform timestamp to UTC if it is in local time 
        self.__convert_local_time_to_UTC__(df, inplace=True, timestamp_column=timestamp_column)
        
        # currently no support of duplicated timestamps 
        assert len(df[df[timestamp_column].duplicated()]) == 0, 'HEAPO.change_resolution(): The data frame contains duplicated timestamps. This is currently not supported and may be the case when multiple households are in one data frame. Please remove duplicates in timestamps.'

        # sort values by timestamp 
        df.sort_values(by=timestamp_column, inplace=True)
            
        # get the columns that should be handled by max / min / mean / sum / first
        cols = df.columns.values
        cols_max = [col for col in cols if '_max_' in col.lower() and col in self.description_weather['VariableName'].values]
        cols_min = [col for col in cols if '_min_' in col.lower() and col in self.description_weather['VariableName'].values]
        cols_sum = [col for col in cols if 'kwh_' in col.lower() or 'kvarh_' in col.lower() or (col in self.description_weather['VariableName'].values and ('_total_' in col.lower() or '_duration_' in col.lower()))]
        cols_mean = [col for col in cols if '_mean_' in col.lower() and col in self.description_weather['VariableName'].values]
        cols_first = [col for col in cols if col not in cols_max+cols_min+cols_sum+cols_mean and col != timestamp_column]
        
        # additional assertions
        for col in cols: 
            if col != timestamp_column:
                assert col in cols_max+cols_min+cols_sum+cols_mean+cols_first, 'HEAPO.change_resolution(): The column {} is missing in the definition of resampling.'.format(col)
        for col in cols_max: 
            assert col not in cols_min+cols_sum+cols_mean+cols_first, 'HEAPO.change_resolution(): The column {} is defined for multiple resampling methods.'.format(col)
        for col in cols_min: 
            assert col not in cols_max+cols_sum+cols_mean+cols_first, 'HEAPO.change_resolution(): The column {} is defined for multiple resampling methods.'.format(col)
        for col in cols_sum:
            assert col not in cols_max+cols_min+cols_mean+cols_first, 'HEAPO.change_resolution(): The column {} is defined for multiple resampling methods.'.format(col)
        for col in cols_mean:
            assert col not in cols_max+cols_min+cols_sum+cols_first, 'HEAPO.change_resolution(): The column {} is defined for multiple resampling methods.'.format(col)
        for col in cols_first:
            assert col not in cols_max+cols_min+cols_sum+cols_mean, 'HEAPO.change_resolution(): The column {} is defined for multiple resampling methods.'.format(col)

        # save previous dates if newly created days should be removed
        if not add_missing_dates:
            previous_dates = df[timestamp_column].dt.date.unique()
        
        # resample data
        # NOTE: unfortunately there is a bug in pandas where resampling does not work correctly with NAN values, which is why the apply method needs to be used if NANs should be skipped, which is slower in computation 
        # NOTE: for more information, check this thread: https://github.com/pandas-dev/pandas/issues/29382 
        concat_dfs = []
        if len(cols_max) > 0:
            concat_dfs.append(df[[timestamp_column]+cols_max].copy().set_index(timestamp_column).resample(resolution).max().reset_index())
        if len(cols_min) > 0:
            concat_dfs.append(df[[timestamp_column]+cols_min].copy().set_index(timestamp_column).resample(resolution).min().reset_index())
        if len(cols_sum) > 0:
            if skipna:
                concat_dfs.append(df[[timestamp_column]+cols_sum].copy().set_index(timestamp_column).resample(resolution).sum().reset_index())
            else: 
                concat_dfs.append(df[[timestamp_column]+cols_sum].copy().set_index(timestamp_column).resample(resolution).apply(lambda x: np.sum(x.values)).reset_index())
        if len(cols_mean) > 0:
            if skipna:
                concat_dfs.append(df[[timestamp_column]+cols_mean].copy().set_index(timestamp_column).resample(resolution).mean().reset_index())
            else: 
                concat_dfs.append(df[[timestamp_column]+cols_mean].copy().set_index(timestamp_column).resample(resolution).apply(lambda x: np.mean(x.values)).reset_index())
        if len(cols_first) > 0:
            concat_dfs.append(df[[timestamp_column]+cols_first].copy().set_index(timestamp_column).resample(resolution).first().reset_index())
        
        # now put the data frames back together 
        assert len(concat_dfs) > 0, 'HEAPO.change_resolution(): No columns to be resampled were found.'
        if len(concat_dfs) > 1: # if there are multiple data frames to be concatenated
            for df_temp in concat_dfs[1:]:
                df_temp.drop(columns=[timestamp_column], inplace=True)
            df = pd.concat(concat_dfs, axis=1)
        else: 
            df = concat_dfs[0]
        
        # restore original order of columns
        assert all(col in df.columns.values for col in cols), 'HEAPO.change_resolution(): Not all columns are available in the resampled data frame. Missing columns: {}'.format([col for col in cols if col not in df.columns.values])
        df = df[cols]

        # premove previously non-existent dates if this is desired
        if not add_missing_dates:
            df = df[df[timestamp_column].dt.date.isin(previous_dates)]
            df.reset_index(drop=True, inplace=True)

        # convert back to local time if this is desired
        if self.use_local_time: 
            df = self.__convert_UTC_to_local_time__(df, inplace=True, timestamp_column=timestamp_column)
        
        return df 
    
    def prepare_groundtruth_from_protocols(self):
        '''
            Prepares relevant ground truth from protocols with respect to training task for heat pump evaluation. 
            NOTE: covers ground truth for predicting heat pump configurations and general heat pump evaluation
            NOTE: does not cover prediction tasks specific to only one type of heat pump 
            NOTE: calculates ground truth for all protocols with known household ID and where household only had one consultation 
            NOTE: however, no further selection of protocols or household is done, i.e., you may want to filter the ground truth afterwards with respect to household types to consider
            Returns: 
                data frame with relevant ground truth derived from the available protocols
        '''

        # create a copy of the protocols that only keeps households with one consultation and where the household ID is known
        df_protocols = self.get_all_protocols()
        df_protocols = df_protocols[df_protocols[self.household_id].notna()]
        df_protocols = df_protocols[~df_protocols[self.household_id].isin(self.get_all_households_with_multiple_protocols())] 

        # select relevant variables 
        variables = [
            'HeatPump_BasicFunctionsOkay',
            'HeatPump_TechnicallyOkay',

            'HeatPump_Installation_CorrectlyPlanned',
            'HeatPump_Installation_IncorrectlyPlanned_Categorization',

            'HeatPump_ElectricityConsumption_Categorization', 

            'HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit',
            'HeatPump_HeatingCurveSetting_Changed',

            'HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit',
            'HeatPump_HeatingLimitSetting_Changed',

            'HeatPump_HeatingLimitSetting_BeforeVisit',
            'HeatPump_HeatingLimitSetting_AfterVisit',

            'HeatPump_NightSetbackSetting_Activated_BeforeVisit',
            'HeatPump_NightSetbackSetting_Activated_AfterVisit',

            'DHW_TemperatureSetting_Categorization',
            'DHW_TemperatureSetting_Changed',
        ]
        # create ground truth data frame for before consultation 
        df_before = df_protocols[[self.household_id] + variables].copy()

        # add a column that indicates that this is the ground truth for before the consultation
        df_before.insert(1, 'AffectsTimePoint', 'before visit')

        # make first three variables binary and the fourth categorical
        df_before[variables[:3]] = df_before[variables[:3]].fillna(True).astype(int).astype(float) # encode boolean variables - assume everything okay if not mentioned otherwise
        df_before['HeatPump_ElectricityConsumption_Categorization'].replace(to_replace={'normal': 1.0, 'rather low': 2.0, 'rather high' : 3.0}, inplace=True) # make variable categorical
        
        # create a version for after consultation and concatenate both
        df_after = df_before.copy()
        df_after['AffectsTimePoint'].replace(to_replace={'before visit' : 'after visit'}, inplace=True)
        df_gt = pd.concat([df_before, df_after], axis=0)
        df_gt.sort_values(by=[self.household_id, 'AffectsTimePoint'], inplace=True, ascending=False)
        df_gt.reset_index(drop=True, inplace=True)

        # now process all data 
        df_gt.loc[(df_gt['AffectsTimePoint'] == 'after visit'), 'HeatPump_BasicFunctionsOkay'] = np.nan # we do not know evaluation of basic functions after the consultation
        df_gt.loc[(df_gt['AffectsTimePoint'] == 'after visit'), 'HeatPump_TechnicallyOkay'] = np.nan # we do not know evaluation of basic functions after the consultation
        df_gt.loc[(df_gt['AffectsTimePoint'] == 'after visit'), 'HeatPump_ElectricityConsumption_Categorization'] = np.nan # we do not know the electricity consumption after the consultation

        # ---------------------------------------
        # handle electricity consumption too high
        # ---------------------------------------

        df_gt.insert(6, 'HeatPump_ElectricityConsumption_TooHigh', 0.0) 
        df_gt.loc[(df_gt['AffectsTimePoint'] == 'before visit') & (df_gt['HeatPump_ElectricityConsumption_Categorization'] == 3.0), 'HeatPump_ElectricityConsumption_TooHigh'] = 1.0 # electricity consumption too high before visit
        df_gt.loc[df_gt['AffectsTimePoint'] == 'after visit', 'HeatPump_ElectricityConsumption_TooHigh'] = np.nan

        # ----------------------------
        # handle installation correctly planned categorization 
        # ----------------------------

        df_gt['HeatPump_Installation_IncorrectlyPlanned_Categorization'].replace(to_replace={'undersized': 2.0, 'oversized': 3.0}, inplace=True)
        df_gt['HeatPump_Installation_IncorrectlyPlanned_Categorization'] = df_gt['HeatPump_Installation_IncorrectlyPlanned_Categorization'].astype(float)
        df_gt['HeatPump_Installation_IncorrectlyPlanned_Categorization'] = df_gt['HeatPump_Installation_IncorrectlyPlanned_Categorization'].fillna(1.0)
        df_gt.rename(columns={'HeatPump_Installation_IncorrectlyPlanned_Categorization': 'HeatPump_Installation_CorrectlyPlanned_Categorization'}, inplace=True)

        # ----------------------------
        # handle heating curve setting
        # ---------------------------- 

        # assume that if heating curve was too high before visit, consultant would have said so, therefore fill NAN with False 
        df_gt['HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit'].fillna(False, inplace=True)

        # same for the change - assume that a change would have been mentioned 
        df_gt['HeatPump_HeatingCurveSetting_Changed'].fillna(False, inplace=True)
        
        # now process changes in heating curve setting 
        # where heating curve setting was too high before visit and was changed - assume that it is not too high afterwards 
        # otherwise it should remain as it was before the visit
        df_gt.loc[(df_gt['AffectsTimePoint'] == 'after visit') & (df_gt['HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit']==True) & (df_gt['HeatPump_HeatingCurveSetting_Changed']==True), 'HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit'] = False # restart for the after visit
        
        # now finally reset the variable name 
        df_gt.rename(columns={'HeatPump_HeatingCurveSetting_TooHigh_BeforeVisit': 'HeatPump_HeatingCurveSetting_TooHigh'}, inplace=True)
        df_gt.drop(columns=['HeatPump_HeatingCurveSetting_Changed'], inplace=True)
        df_gt['HeatPump_HeatingCurveSetting_TooHigh'] = df_gt['HeatPump_HeatingCurveSetting_TooHigh'].astype(int).astype(float) # make binary

        # ----------------------------
        # handle heating limit setting
        # ----------------------------

        # create one common column for actual heating limit 
        df_gt.loc[df_gt['AffectsTimePoint'] == 'after visit', 'HeatPump_HeatingLimitSetting_BeforeVisit'] = df_gt.loc[df_gt['AffectsTimePoint'] == 'after visit', 'HeatPump_HeatingLimitSetting_AfterVisit']
        df_gt.rename(columns={'HeatPump_HeatingLimitSetting_BeforeVisit': 'HeatPump_HeatingLimitSetting'}, inplace=True)
        df_gt.drop(columns=['HeatPump_HeatingLimitSetting_AfterVisit'], inplace=True)

        # assume that if heating limit setting was too high before visit, consultant would have said so, therefore fill NAN with False
        df_gt['HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit'].fillna(False, inplace=True)

        # same for the change - assume that a change would have been mentioned
        df_gt['HeatPump_HeatingLimitSetting_Changed'].fillna(False, inplace=True)

        # now process changes in heating limit setting
        # where heating limit setting was too high before visit and was changed - assume that it is not too high afterwards
        # otherwise it should remain as it was before the visit
        df_gt.loc[(df_gt['AffectsTimePoint'] == 'after visit') & (df_gt['HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit']==True) & (df_gt['HeatPump_HeatingLimitSetting_Changed']==True), 'HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit'] = False # restart for the after visit

        # now finally reset the variable name
        df_gt.rename(columns={'HeatPump_HeatingLimitSetting_TooHigh_BeforeVisit': 'HeatPump_HeatingLimitSetting_TooHigh'}, inplace=True)
        df_gt.drop(columns=['HeatPump_HeatingLimitSetting_Changed'], inplace=True)
        df_gt['HeatPump_HeatingLimitSetting_TooHigh'] = df_gt['HeatPump_HeatingLimitSetting_TooHigh'].astype(int).astype(float) # make binary

        # ----------------------------
        # handle night setback setting
        # ----------------------------

        # assume that if night setback setting was activated before visit, consultant would have said so, therefore fill NAN with False
        df_gt['HeatPump_NightSetbackSetting_Activated_BeforeVisit'].fillna(False, inplace=True)
        df_gt.loc[df_gt['AffectsTimePoint'] == 'after visit', 'HeatPump_NightSetbackSetting_Activated_BeforeVisit'] = df_gt.loc[df_gt['AffectsTimePoint'] == 'after visit', 'HeatPump_NightSetbackSetting_Activated_AfterVisit']
        df_gt.rename(columns={'HeatPump_NightSetbackSetting_Activated_BeforeVisit': 'HeatPump_NightSetbackSetting_Activated'}, inplace=True)
        df_gt.drop(columns=['HeatPump_NightSetbackSetting_Activated_AfterVisit'], inplace=True)
        df_gt['HeatPump_NightSetbackSetting_Activated'].replace(to_replace={False: 0.0, True: 1.0})
        df_gt['HeatPump_NightSetbackSetting_Activated'] = df_gt['HeatPump_NightSetbackSetting_Activated'].astype(float) # make binary

        # ----------------------------
        # handle DHW setting
        # ----------------------------

        # assume that where ever it was changed it should be normal afterwards 
        df_gt.loc[(df_gt['AffectsTimePoint'] == 'after visit') & (df_gt['DHW_TemperatureSetting_Changed']==True), 'DHW_TemperatureSetting_Categorization'] = 'normal'
        df_gt.drop(columns=['DHW_TemperatureSetting_Changed'], inplace=True)
        df_gt['DHW_TemperatureSetting_Categorization'].replace(to_replace={'normal': 1.0, 'too low': 2.0, 'too high' : 3.0}, inplace=True) # make variable categorical
        df_gt.insert(len(df_gt.columns.values)-1, 'DHW_TemperatureSetting_TooHigh', 0.0)
        df_gt.loc[df_gt['DHW_TemperatureSetting_Categorization'] == 3.0, 'DHW_TemperatureSetting_TooHigh'] = 1.0

        return df_gt
    
    def prepare_features_from_protocols(self): 
        '''
            Prepares relevant context from protocols with respect to training task for heat pump evaluation. 
            NOTE: this functionality does not consider all available protocol variables and can be extended upon individual needs!!! 
            NOTE: covers a selection of static variables about heat pump, heating system and building that remain unchanged over time
            NOTE: already performs coding of categorical variables and one-hot-encoding where necessary, such that features can be used directly
            NOTE: calculates this context for all protocols with known household ID and where household only had one consultation 
            NOTE: however, no further selection of protocols or household is done, i.e., you may want to filter the context afterwards with respect to household types to consider
            Returns: 
                data frame with relevant context derived from the available protocols
        '''
        
        # create a copy of the protocols that only keeps households with one consultation and where the household ID is known
        df_protocols = self.get_all_protocols()
        df_protocols = df_protocols[df_protocols[self.household_id].notna()]
        df_protocols = df_protocols[~df_protocols[self.household_id].isin(self.get_all_households_with_multiple_protocols())]

        # select relevant variables
        relevant_variables = [ 
            'Building_Type', # needs filling of NAN as single family home and then categorical encoding 
            'Building_HousingUnits', # needs filling of NAN with 1 
            'Building_ConstructionYear_Interval', # needs categorical encoding
            'Building_Renovated_Windows', # will be used to calculate single variable for state of renovation
            'Building_Renovated_Roof', # will be used to calculate single variable for state of renovation
            'Building_Renovated_Walls', # will be used to calculate single variable for state of renovation
            'Building_Renovated_Floor', # will be used to calculate single variable for state of renovation
            'Building_FloorAreaHeated_Total', 
            'Building_Residents',
            'Building_PVSystem_Available', # fill NAN with False and then encode as int 
            'Building_ElectricVehicle_Available', # fill NAN with False and then encode as int 
            'HeatPump_Installation_Type', # needs categorical encoding
            'HeatPump_Installation_HeatingCapacity', 
            'HeatDistribution_System_Radiators', # fill NAN with False and then encode as int
            'HeatDistribution_System_FloorHeating', # fill NAN with False and then encode as int
            'HeatDistribution_System_ThermostaticValve', # fill NAN with False and then encode as int
            'DHW_Production_ByHeatPump', # fill NAN with False and then encode as int
            'DHW_Production_ByElectricWaterHeater', # fill NAN with False and then encode as int
            'DHW_Production_BySolar', # fill NAN with False and then encode as int
            'DHW_Production_ByHeatPumpBoiler', # fill NAN with False and then encode as int
            'DHW_Circulation_ByTraceHeating', # fill NAN with False and then encode as int
            'DHW_Circulation_ByCirculationPump', # fill NAN with False and then encode as int
            'DHW_Circulation_SwitchedByTimer', # fill NAN with False and then encode as int
            'DHW_Sterilization_Available' # fill NAN with False and then encode as int
        ]
        df_protocols = df_protocols[[self.household_id, 'Visit_Year', 'Visit_Date'] + relevant_variables]

        # calculate single variable to indicate state of renovation 
        df_protocols['Building_Renovated'] = False 
        df_protocols.loc[(df_protocols['Building_Renovated_Windows'] == True) | (df_protocols['Building_Renovated_Roof'] == True) | (df_protocols['Building_Renovated_Walls'] == True) | (df_protocols['Building_Renovated_Floor'] == True), 'Building_Renovated'] = True
        df_protocols.drop(columns=['Building_Renovated_Windows', 'Building_Renovated_Roof', 'Building_Renovated_Walls', 'Building_Renovated_Floor'], inplace=True)
        df_protocols.insert(6, 'Building_Renovated', df_protocols.pop('Building_Renovated'))

        # handle boolean columns 
        bool_cols = [
            'Building_Renovated', 
            'Building_PVSystem_Available',
            'Building_ElectricVehicle_Available',
            'HeatDistribution_System_Radiators', 
            'HeatDistribution_System_FloorHeating', 
            'HeatDistribution_System_ThermostaticValve', 
            'DHW_Production_ByHeatPump', 
            'DHW_Production_ByElectricWaterHeater', 
            'DHW_Production_BySolar', 
            'DHW_Production_ByHeatPumpBoiler', 
            'DHW_Circulation_ByTraceHeating', 
            'DHW_Circulation_ByCirculationPump',
            'DHW_Circulation_SwitchedByTimer',
            'DHW_Sterilization_Available'
        ]
        for col in bool_cols: 
            df_protocols[col] = df_protocols[col].fillna(False).astype(int).astype(float)

        # handle other columns 
        df_protocols['Building_Residents'] = df_protocols['Building_Residents'].astype(float)
        df_protocols['Building_Type'] = df_protocols['Building_Type'].fillna('single family house').replace(to_replace={'single family house':1.0, 'multi family house':2.0, 'other':3.0}).astype(float)
        df_protocols['Building_HousingUnits'] = df_protocols['Building_HousingUnits'].fillna(1.0).astype(float)
        to_replace = {'< 1975' : 1.0, '1976 - 80' : 2.0, '1981 - 85' : 3.0, '1986 - 90' : 4.0, '1991 - 95' : 5.0, '1995 - 00' : 6.0, '2000 - 10' : 7.0, '> 2010' : 8.0}
        df_protocols['Building_ConstructionYear_Interval'] = df_protocols['Building_ConstructionYear_Interval'].replace(to_replace=to_replace).astype(float)
        df_protocols['HeatPump_Installation_Type'] = df_protocols['HeatPump_Installation_Type'].replace(to_replace={'air-source':1.0, 'ground-source':2.0, 'water-source':3.0}).astype(float)
        
        df_protocols.reset_index(drop=True, inplace=True)
        return df_protocols
    
    def plot_heatmap(self, df:pd.DataFrame, value_column:str, timestamp_column:str=None, ax=None, hour_interval:int=1, vmin:float=None, vmax:float=None, cmap='viridis', cbar:bool=False, cbarlabel='Energy (kWh)', fontsize:int=14, figsize=(12, 8)): 
        '''
            Helper function to plot a heat map on a given axis.
            NOTE: currently only supports heat maps for time series data with a temporal resolution of less than one day and equal to or above 1 minute
            NOTE: no duplicates in timestamp allowed - i.e., data frame should only contain a single household
            Args:
                df: dataframe containing the data to plot
                value_column: column name with values to create the heat map data for - must be numeric 
                timestamp_column: column name with the timestamp - must be of type datetime - if None, HEAPO.timestamp is used
                ax: axis to plot the heat map on - if None, will create figure and axis
                hour_interval: (int) interval for the y-axis labels in hours
                vmin: (float) minimum value for the color map
                vmax: (float) maximum value for the color map
                cmap: (string) name of the color map to use
                cbar: (bool) if True, a color bar is added to the plot
                cbarlabel: (string) label for the color bar
                fontsize: (int) font size for the labels
                figsize: (tuple) size of the figure to create if ax is None
            Returns:
                dictionary with the following keys and values: 
                    - mesh: (matplotlib.collections.QuadMesh) the heat map object
                    - cbar: (matplotlib.colorbar.Colorbar) the color bar object - None if cbar is False
                    - df_heatmap: (dataframe) the 2D representation of the data
                    - ax: (matplotlib.axes._subplots.AxesSubplot) the axis object
                    - fig: (matplotlib.figure.Figure) the figure object
        '''

        if timestamp_column is None:
            timestamp_column = self.timestamp
        assert isinstance(hour_interval, int) and hour_interval > 0, 'HEAPO.plot_heatmap(): The argument hour_interval must be an integer greater than zero.'
        if fontsize is not None:
            assert isinstance(fontsize, int) and fontsize > 0, 'HEAPO.plot_heatmap(): The argument fontsize must be an integer greater than zero.'

        # currently no support of duplicated timestamps 
        assert len(df[df[timestamp_column].duplicated()]) == 0, 'HEAPO.plot_heatmap(): The data frame contains duplicated timestamps. This is currently not supported and may be the case when multiple households are in one data frame. Please remove duplicates in timestamps.'

        # get temporal resolution of the data 
        time_delta_minutes = self.__infer_temporal_resolution_in_minutes__(df, timestamp_column)

        # check for allowed temporal resolution for heat map plots (only resolutions below one day and equal to or above 15 min are allowed)
        assert time_delta_minutes < 23.0*60.0, 'HEAPO.plot_heatmap(): Cannot create heamap data because the current resolution of the data is greater than one day. Current resolution in minutes: {}'.format(time_delta_minutes)
        assert time_delta_minutes >= 1.0, 'HEAPO.plot_heatmap(): The current resolution of the data is smaller than 1 minute. This is not supported.'

        # get heat map data 
        df_heatmap = self.__create_heatmap_data__(df, value_column, timestamp_column=timestamp_column, add_missing_times=True)

        if vmin is not None:
            assert isinstance(vmin, (int, float)), 'HEAPO.plot_heatmap(): The argument cmin must be of type int or float.'
        if vmax is not None:
            assert isinstance(vmax, (int, float)), 'HEAPO.plot_heatmap(): The argument cmax must be of type int or float.'

        # potentially create figure 
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            flag_created = True
        else: 
            fig = ax.get_figure()
            flag_created = False

        # plot heat map
        df_heatmap.index = pd.to_datetime('1970-01-01T'+df_heatmap.index.astype(str))
        timerange = df_heatmap.index
        mesh = ax.pcolormesh(df_heatmap.columns.astype('datetime64[ns]'), timerange, df_heatmap.values, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

        # make sure that time 00:00 is at the top
        ax.invert_yaxis() 

        # x-axis: date formatting 
        if len(df_heatmap.columns) > 150: # if more than 150 days of data --> use month and year
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        else: # otherwise use day and month
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_minor_locator(mdates.DayLocator())

        # y-axis: time formatting
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:00")) # TODO: Use '%-H' for linux, '%#H' for windows to remove leading zero
        ax.yaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
        ax.yaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, hour_interval)))

        # fill background with grey for NAN values 
        ax.set_facecolor('grey')
        
        # add grid 
        ax.grid(alpha=0.3)

        # add labels 
        ax.set_xlabel('Date', fontsize=fontsize, fontweight='bold', labelpad=10)
        timelabel = 'Time (UTC)' if not self.use_local_time else 'Time (Local - Europe / Zurich)'
        ax.set_ylabel(timelabel, fontsize=fontsize, fontweight='bold', labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=10)

        # add color bar if necessary
        if cbar: 
            self.plot_cbar_on_ax(mesh, ax, cax=None, fontsize=fontsize, cbarlabel=cbarlabel, labelpad=10, orientation='vertical', pad=0.02)
        else: 
            cbar = None

        # tight layout if figure was created 
        if flag_created:
            fig.tight_layout()

        return {'mesh': mesh, 'cbar': cbar, 'df_heatmap': df_heatmap, 'ax' : ax, 'fig' : fig}
    
    def plot_cbar_on_ax(self, mappable, ax=None, cax=None, fontsize:int=None, cbarlabel:str='Energy (kWh)', labelpad:int=10, **kwargs): 
        ''' 
            Helper function to add a color bar to a given axis. 
            See documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html 
            Args: 
                mappable: mappable object described by colorbar
                cax: Axes into which the colorbar will be drawn. If None, then a new Axes is created and the space for it will be stolen from the Axes(s) specified in ax.
                ax: The one or more parent Axes from which space for a new colorbar Axes will be stolen. This parameter is only used if cax is not set.
                fontsize: font size for the labels 
                cbarlabel: label for the color bar 
                labelpad: padding for the label 
                **kwargs: additional keyword arguments for the color bar 
            Returns: 
                color bar object
        '''
        if ax is not None:
            assert isinstance(ax, plt.Axes), 'HEAPO.plot_cbar_on_ax(): The argument ax must be of type matplotlib.axes._subplots.AxesSubplot.'
        try: 
            cbar = plt.colorbar(mappable, cax=cax, ax=ax, **kwargs)
            if cbarlabel is not None:
                cbar.set_label(cbarlabel, fontsize=fontsize, fontweight='bold', labelpad=labelpad)
            cbar.ax.tick_params(labelsize=fontsize, pad=10)
        except Exception as e: 
            if self.warnings: 
                warnings.warn('HEAPO.plot_cbar_on_ax(): Could not add color bar to the axis. Error: {}'.format(e))
            cbar = None
        return cbar
    
    def plot_consultation_range_on_ax(self, ax, df:pd.DataFrame, report_date, timestamp_column:str=None, fontsize:int=None, annotate_before_after:bool=True, color_before:str=COLOR_PURPLE, color_during:str=COLOR_DARK, color_after:str=COLOR_YELLOW):
        '''
            Plots a range around a consultation date on a given axis.
            Args:
                ax: axis to plot the range on
                df: dataframe containing the data to plot
                report_date: date of the consultation - either datetime.date object or string in format 'YYYY-MM-DD'
                timestamp_column: column name with the timestamp - if None, HEAPO.timestamp is used
                fontsize: font size for the annotations
                annotate_before_after: if True, the periods before and after the consultation are annotated
                color_before: color for the period before the consultation
                color_during: color for the period of the consultation
                color_after: color for the period after the consultation
            Returns:
                ax: the axis object 
        '''
        assert isinstance(ax, plt.Axes), 'HEAPO.plot_consultation_range_on_ax(): The argument ax must be of type matplotlib.axes._subplots.AxesSubplot.'
        if timestamp_column is None:
            timestamp_column = self.timestamp
        assert isinstance(annotate_before_after, bool), 'HEAPO.plot_consultation_range_on_ax(): The argument annotate_before_after must be of type bool.'
        assert timestamp_column in df.columns.values, 'HEAPO.plot_consultation_range_on_ax(): The data frame does not contain a column named "{}".'.format(timestamp_column)
        assert pd.api.types.is_datetime64_any_dtype(df[timestamp_column]), 'HEAPO.plot_consultation_range_on_ax(): The data frame column "{}" is not of type datetime.'.format(timestamp_column)

        # resample to daily resolution - also get days that may be missing in the data frame
        df = df.copy()[[timestamp_column]]
        df.set_index(timestamp_column, inplace=True)
        df = df.resample('1D').mean().sort_values(by=timestamp_column, ascending=True).reset_index()
        df['Date'] = df[timestamp_column].dt.date

        # transform date of the consultation to datetime 
        if not isinstance(report_date, datetime.date):
            report_date = pd.to_datetime(report_date, format='%Y-%m-%d').date()

        # get number of dates before, during, and after
        vals_before = len(df[df['Date'] < report_date][timestamp_column].dt.date.unique())
        vals_after = len(df[df['Date'] > report_date][timestamp_column].dt.date.unique())
        vals_during = len(df[df['Date'] == report_date][timestamp_column].dt.date.unique())
        vals_total = vals_before + vals_after + vals_during
        assert vals_total >= 1, 'HEAPO.plot_consultation_range_on_ax(): The data frame does not contain any data around the consultation date.'

        # prepare ax 
        ax.set_xlim(0, vals_total)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1)

        # plot before period
        if vals_before >= 1: 
            ax.fill_between(range(0, vals_before+1), 1.0, color=color_before, alpha=0.6)
            if annotate_before_after:
                ax.annotate('Before', xy=(0.5*(vals_before+1), 0.35), fontsize=fontsize)

        # plot during period
        if vals_during >= 1: 
            ax.fill_between([vals_before, vals_before+1], 1.0, color=color_during, alpha=1.0, linewidth=2.0)
    
        # plot after period
        if vals_after >= 1: 
            ax.fill_between(range(vals_before+vals_during, vals_total+1), 1.0, color=color_after, alpha=0.6)
            if annotate_before_after:
                ax.annotate('After', xy=(vals_before+0.5*(vals_total-vals_before+1), 0.35), fontsize=fontsize)
            
        return ax