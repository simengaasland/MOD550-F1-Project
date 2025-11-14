import fastf1 as f1
import os
import numpy as np
import pandas as pd

class NameTBD:
    def __init__(self):
        cache_dir = "./f1_cache"
        os.makedirs(cache_dir, exist_ok=True)
        f1.Cache.enable_cache("./f1_cache")

    def get_practice_laps(self, year, gp):
        '''
        Collects FP1, FP2 and FP3 data from
        the FastF1 API.

        Input
        ------
        gp: string
            The Grand Prix to collect data from (e.g. 'Italian Grand Prix').

        year: int
            The year the Grand Prix took place (e.g. 2024).

        Output
        ------
        DataFrames
            DataFrame containing FP1, FP2 & FP3.
        '''
        fp_sessions = []
        fp_weather = []
        fp_fastest_lap_sec = np.nan

        for fp in ['FP1', 'FP2', 'FP3']:
            #Try getting FP session from API
            try:
                session = f1.get_session(year = year, gp = gp, identifier = fp)
                session.load()
                fp_sessions.append(session.laps)
                fp_weather.append(session.weather_data)

            #Handling if the GP did not have FP session
            except:
                print(f'No {fp} session found for {gp} {year}')
                fp_sessions.append(pd.DataFrame())
                fp_weather.append(pd.DataFrame())

        #Combining list of DataFrames into one DataFrame
        fp_sessions = pd.concat(fp_sessions, ignore_index=True)
        fp_weather = pd.concat(fp_weather, ignore_index=True)

        if not fp_sessions.empty:
            fp_fastest_lap = fp_sessions['LapTime'].min()
            fp_fastest_lap_sec = fp_fastest_lap.total_seconds()

        return fp_sessions, fp_fastest_lap_sec, fp_weather



    def get_fastest_race_lap_and_pos(self, year, gp):
        '''
        Function that returns the fastest lap of the race

        Input
        ------
        gp: string
            The Grand Prix to collect data from (e.g. 'Italian Grand Prix').

        year: int
            The year the Grand Prix took place (e.g. 2024).

        Output
        ------
        DataFrame???? Float
            A DataFrame with ...???
        '''
        #Empty DataFrame
        position_race_data = pd.DataFrame()
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
            return np.nan, position_race_data, race_weather_data

        try:
            #Find the fastest lap of the race
            df_race_fastest_lap = race_session.laps.pick_fastest()

            #Finds result of race
            df_race_results = race_session.results

            #Get weather data
            df_race_weather = race_session.weather_data
        except:
            return np.nan, position_race_data, race_weather_data

        #Convert laptime from DateTime to float (seconds)
        #Races like the Belgian GP in 2021 returns NaN,
        #since every lap of the race was completed under
        #a safetycar, and therefore has no fastest lap.
        try:
            race_fastest_lap = df_race_fastest_lap['LapTime'].total_seconds()

            position_race_data['FasterThanTeammateRace'] = (df_race_results['Position'] == df_race_results.groupby(['TeamName'])
                                         ['Position'].transform('min')).astype(float)
            position_race_data['PointFinishRace'] = (df_race_results['Position'] <= 10).astype(int)

            race_weather_data = {'TrackTempAvgRace': df_race_weather['TrackTemp'].mean(),
                                 'AirTempAvgRace': df_race_weather['AirTemp'].mean(),
                                 'RainAvgRace':  df_race_weather['Rainfall'].mean()}

            return race_fastest_lap, position_race_data, race_weather_data
        except:
            return np.nan, position_race_data, race_weather_data


    def get_fastest_laps(self, year, gp):
        '''
        Collects FP1, FP2 and FP3 data from
        the FastF1 API and cleans it. The
        collect the data from all three
        practice sessions. For all session
        it removes deleted laps and
        laps times with NaN-value. For
        each driver the function then finds
        the fastest lap preformed by that
        driver for all three session. At
        last it combines all drivers fastest
        lap for all three sessions.

        Input
        ------
        gp: string
            The Grand Prix to collect data from (e.g. 'Italian Grand Prix').

        year: int
            The year the Grand Prix took place (e.g. 2024).

        Output
        ------
        DataFrame
            A DataFrame with driver number, session and lap
            time in seconds
        '''
        fp_sessions, fp_fastest_lap_sec, fp_weather = self.get_practice_laps(year, gp)

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
                fastest_lap_of_race, position_race_data, weather_race_data = self.get_fastest_race_lap_and_pos(year, gp)
                df_fp_data = self.get_fastest_laps(year, gp)
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
        data.to_csv('F1_data_new_2025.csv', index=False)

    def faster_then_teammate_FP(self, data):
        '''
        Adds 'FasterThenTeammate' column to dataframe
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

if __name__ == '__main__':
    #years = [2024, 2023, 2022, 2021, 2019]
    years = [2025]
    obj = NameTBD()
    obj.get_data_from_api(years)
