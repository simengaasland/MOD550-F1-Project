import fastf1 as f1
import os
import numpy as np
import pandas as pd

class DataAcquisition:
    def __init__(self):
        cache_dir = "./f1_cache"
        os.makedirs(cache_dir, exist_ok=True)
        f1.Cache.enable_cache("./f1_cache")

    def get_practice_features_and_weather(self, year, gp):
        '''
        Collects FP1, FP2 and FP3 data for a given Grand Prix using the
        FastF1 API.

        This function loads FP1, FP2 and FP3 sessions. When a session is
        missing (e.g. sprint weekends), it inserts an empty DataFrame.
        It also returns the fastest lap for all sessions in seconds.
        Lastly, it returns a DataFrame with the weather data for all
        sessions.

        Input
        ------
        gp: string
            The Grand Prix to collect data from (e.g. 'Italian Grand Prix').

        year: int
            The year the Grand Prix took place (e.g. 2024).

        Output
        ------
        tuple:
            fp_sessions (DataFrame):
                Combined Dataframe of all FP1, FP2 and FP3 laps.
            fp_fastest_lap_sec (float):
                Fastest lap of all FP sessions in seconds.
            fp_weather (DataFrame):
                Combinded weather data for all FP sessions.
        '''
        fp_sessions_list = []
        fp_weather_list = []
        fp_fastest_lap_sec = np.nan

        fp_sessions = ['FP1', 'FP2', 'FP3']

        for fp in fp_sessions:
            #Try getting FP session from API
            try:
                session = f1.get_session(year = year, gp = gp, identifier = fp)
                session.load()

                fp_sessions_list.append(session.laps)
                fp_weather_list.append(session.weather_data)

            #Handling if the GP did not have FP session
            except:
                print(f'No {fp} session found for {gp} {year}')
                fp_sessions_list.append(pd.DataFrame())
                fp_weather_list.append(pd.DataFrame())

        #Combining list of DataFrames into one DataFrame
        fp_sessions = pd.concat(fp_sessions_list, ignore_index=True)
        fp_weather = pd.concat(fp_weather_list, ignore_index=True)

        if not fp_sessions.empty:
            fp_fastest_lap = fp_sessions['LapTime'].min()
            fp_fastest_lap_sec = fp_fastest_lap.total_seconds()

        return fp_sessions, fp_fastest_lap_sec, fp_weather



    def get_race_features_and_weather(self, year, gp):
        '''
        Retrieves the fastest race lap, finishing positions features,
        and average race-day weather.

        This method loads the race session of a specified Grand Prix
        and extracts the fastest race lap then converts it to seconds.
        For races like the Belgian GP in 2021 where there is no fastest
        lap, it returns NaN.

        Furthermore, it collects the race results and calculates
        "FasterThanTeammate" and "PointFinishRace". This is done
        here to limit the amount of API calls.

        Lastly, it collects race-day weather data and calculated the
        average, it is then stored in a dictionary.

        Input
        ------
        gp: string
            The Grand Prix to collect data from (e.g. 'Italian Grand Prix').

        year: int
            The year the Grand Prix took place (e.g. 2024).

        Output
        ------
        tuple:
            race_fastest_lap_sec (float):
                Fastest lap of the race in seconds.
            race_position_data (DataFrame):
                DataFrame with:
                    - FasterThanTeammate (int)
                    - PointFinishRace (int)
            race_weather_data (dictionary):
                Dictionary with:
                    - TrackTempAvgRace (float)
                    - AirTempAvgRace (float)
                    - RainAvgRace (float)
        '''
        #Initialize
        race_position_data = pd.DataFrame()
        race_weather_data = {
            'TrackTempAvgRace': np.nan,
            'AirTempAvgRace': np.nan,
            'RainAvgRace': np.nan
        }

        #Collecting and loading Race session
        try:
            race_session = f1.get_session(year = year, gp = gp, identifier = 'Race')
            race_session.load()
        except:
            print(f'Could not load race session for {gp} {year}')
            return np.nan, race_position_data, race_weather_data

        try:
            #Find the fastest lap of the race
            df_race_fastest_lap = race_session.laps.pick_fastest()

            #Finds result of race
            df_race_results = race_session.results

            #Get weather data
            df_race_weather = race_session.weather_data
        except:
            return np.nan, race_position_data, race_weather_data

        try:
            #Convert laptime from DateTime to float (seconds)
            race_fastest_lap = df_race_fastest_lap['LapTime'].total_seconds()

            #Faster than teamate in race
            race_position_data['FasterThanTeammateRace'] = (df_race_results['Position'] == df_race_results.groupby(['TeamName'])
                                         ['Position'].transform('min')).astype(int)

            #Finish in point (top 10 finish)
            race_position_data['PointFinishRace'] = (df_race_results['Position'] <= 10).astype(int)

            #Weather data
            race_weather_data = {'TrackTempAvgRace': df_race_weather['TrackTemp'].mean(),
                                 'AirTempAvgRace': df_race_weather['AirTemp'].mean(),
                                 'RainAvgRace':  df_race_weather['Rainfall'].mean()}

            return race_fastest_lap, race_position_data, race_weather_data
        except:
            return np.nan, race_position_data, race_weather_data

    def faster_then_teammate_FP(self, data):
        '''
        Adds 'FasterThenTeammate' column to dataframe.

        Input
        ------
        data: DataFrame
            An almost complete dataset.

        Output
        ------
        DataFrame
            Same DataFrame as input with a added
            'FasterThanTeammateFP' column.
        '''
        data['FasterThanTeammateFP'] = (
            (data['FastestFPLap'] == data.groupby(['Year','GP', 'Team'])['FastestFPLap']
            .transform('min'))
            .astype(float)
            )

        #Set NaN where only one teammate had a valid lap
        only_one_driver = data.groupby(['Year','GP','Team'])['DriverNumber'].transform('count') == 1
        data.loc[only_one_driver, 'FasterThanTeammateFP'] = np.nan

        return data


    def create_skeleton_dataset(self, year, gp):
        '''
        Collects FP1, FP2 and FP3 data from the FastF1 API and cleans it.
        For all session it removes deleted laps and laps times with NaN-value.
        For each driver the method then finds the fastest lap preformed by that
        driver for all three session, and combines all drivers fastest lap for
        all three sessions.

        The method also calculates the mean of all the weather data.

        Furthermore, the method calculated the aggregated statistics for all push
        laps. Push laps here are all laps that are within 2 seconds of the
        drivers fastest lap.

        Finally, the method adds all this in a DataFrame along with DriverNumber,
        Team and GP. This DataFrame is the skeleton of the final dataset.

        Input
        ------
        gp: string
            The Grand Prix to collect data from (e.g. 'Italian Grand Prix').

        year: int
            The year the Grand Prix took place (e.g. 2024).

        Output
        ------
        DataFrame
            A DataFrame with:
                - DriverNumber (string)
                - Team (string)
                - FastestFPLap (float)
                - MeanFPLaps (float)
                - StdFPLaps (float)
                - DeltaBestFPLap (float)
                - TrackTempAvgFP (float)
                - AirTempAvgFP (float)
                - RainAvgFP (float)
                - GP (string)

        '''
        fp_sessions, fp_fastest_lap_sec, fp_weather = self.get_practice_features_and_weather(year, gp)

        #Empty dataframe with columns for driver and their fastest lap
        valid_driver_fastest_lap = pd.DataFrame(columns=['DriverNumber',
                                                         'Team',
                                                         'FastestFPLap',
                                                         'MeanFPLaps',
                                                         'StdFPLaps',
                                                         'DeltaBestFPLap',
                                                         'TrackTempAvgFP',
                                                         'AirTempAvgFP',
                                                         'RainAvgFP',
                                                         'GP'])

        #If there if no FP session
        if fp_sessions.empty:
            return valid_driver_fastest_lap

        #Collect all drivers that have completed a lap in FP
        drivers = fp_sessions['DriverNumber'].unique()

        #Collect weather data
        track_temp_avg = fp_weather['TrackTemp'].mean()
        air_temp_avg = fp_weather['AirTemp'].mean()
        rain_avg = fp_weather['Rainfall'].mean()

        for driver in drivers:
            #Collect all laps completed by driver
            driver_laps = fp_sessions[fp_sessions['DriverNumber'] == driver]

            #Find drivers team
            driver_team = driver_laps['Team'].unique()[0]

            #Remove deleted laps
            valid_driver_laps = driver_laps[driver_laps['Deleted'].astype(bool) == False]

            #Remove NaN lap times
            valid_driver_laps = valid_driver_laps.dropna(subset=['LapTime'])

            #Temp list to store laps in total seconds
            driver_lap_times_sec = []

            #Converting drivers laps to seconds
            driver_number_of_laps = len(valid_driver_laps)

            if driver_number_of_laps != 0:
                for j in range(driver_number_of_laps):
                    driver_lap_times_sec.append(
                        valid_driver_laps.iloc[j]['LapTime'].total_seconds()
                        )

            #If driver has not completed a lap in FP laptime is set to NaN
            else:
                driver_lap_times_sec.append(np.nan)

            #Sort drivers lap times. Fastest lap is at [0]
            driver_lap_times_sec.sort()
            driver_fastest_lap = driver_lap_times_sec[0]

            #How far is driver from fastest lap
            delta_best_lap = fp_fastest_lap_sec - driver_fastest_lap

            #Including only push laps to reduce noice
            driver_push_lap_times_sec = []

            for lap_time in driver_lap_times_sec:
                if abs(lap_time - driver_fastest_lap) < 2.0:
                    driver_push_lap_times_sec.append(lap_time)

            #Update number of valid laps
            driver_number_of_push_laps = len(driver_push_lap_times_sec)

            #Calculate mean and std of all FP sessions
            error_squared = 0
            driver_laps_sum = 0

            if driver_number_of_push_laps == 0:
                driver_lap_std = np.nan
                driver_lap_mean = np.nan

            else:
                for lap in driver_push_lap_times_sec:
                    driver_laps_sum += lap
                driver_lap_mean = driver_laps_sum / driver_number_of_push_laps

                for lap in driver_push_lap_times_sec:
                    error_squared += (lap - driver_lap_mean)**2
                driver_lap_std = np.sqrt(error_squared / driver_number_of_push_laps)

            #Add to dataframe
            valid_driver_fastest_lap.loc[len(valid_driver_fastest_lap)] = [driver,
                                                                           driver_team,
                                                                           driver_fastest_lap,
                                                                           driver_lap_mean,
                                                                           driver_lap_std,
                                                                           delta_best_lap,
                                                                           track_temp_avg,
                                                                           air_temp_avg,
                                                                           rain_avg,
                                                                           gp]

        return valid_driver_fastest_lap

    def get_data_from_api(self, years):
        '''
        The method takes a list of seasons and for each season
        gets the schedule. With this schedule it find every
        GP that took place that year. It then goes through all
        the GPs. For every GP it collect the data for the final
        dataset. Once finished the final dataset get written as
        a .csv file with the name 'F1_data.csv'. The columns of
        this file is
            - DriverNumber
            - Team
            - FastestFPLap
            - MeanFPLaps
            - StdFPLaps
            - DeltaBestFPLap
            - TrackTempAvgFP
            - AirTempAvgFP
            - RainAvgFP
            - GP
            - FastestLapRace
            - Year
            - TrackTempAvgRace
            - AirTempAvgRace
            - RainAvgRace
            - FasterThanTeammateRace
            - PointFinishRace
            - FasterThanTeammateFP

        Input
        ------
        years: list[int]
            List of seasons to collect (e.g. [2024, 2023, 2022]).

        '''

        #Empty list to collect all the DataFrames
        list_of_data = []

        #Collecting schedule for entire season, excluding testing sessions
        for year in years:
            try:
                event_schedule = f1.get_event_schedule(year, include_testing = False)
            except:
                print(f'Could not not schedule for year {year}')
                continue

            #Getting a list of all gps (['Italian Grand Prix', ...])
            gps = event_schedule['EventName'].tolist()

            #Adding fastest FP and Race laps and year gp took place
            for gp in gps:
                fastest_lap_of_race, position_race_data, weather_race_data = self.get_race_features_and_weather(year, gp)
                df_fp_data = self.create_skeleton_dataset(year, gp)
                df_fp_data['FastestLapRace'] = fastest_lap_of_race
                df_fp_data['Year'] = year
                df_fp_data['TrackTempAvgRace'] = weather_race_data['TrackTempAvgRace']
                df_fp_data['AirTempAvgRace'] = weather_race_data['AirTempAvgRace']
                df_fp_data['RainAvgRace'] = weather_race_data['RainAvgRace']
                df_fp_data = df_fp_data.join(position_race_data, on='DriverNumber')
                list_of_data.append(df_fp_data)

        #Combining list of DataFrames into one DataFrame
        data = pd.concat(list_of_data, ignore_index=True)

        #Removes NaN values for set for drivers who has not completed a FP lap
        data = data.dropna(subset=['FastestFPLap'])

        #Adds fasten than teammate column
        data = self.faster_then_teammate_FP(data)

        #Convert DataFrame to csv file
        data.to_csv('F1_data.csv', index=False)

if __name__ == '__main__':
    #years = [2024, 2023, 2022, 2021, 2019]
    years = [2025]
    obj = NameTBD()
    obj.get_data_from_api(years)