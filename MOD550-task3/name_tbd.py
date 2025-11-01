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

        for fp in ['FP1', 'FP2', 'FP3']:
            #Try getting FP session from API
            try:
                session = f1.get_session(year = year, gp = gp, identifier = fp)
                session.load()
                fp_sessions.append(session.laps)

            #Handling if the GP did not have FP session
            except:
                print(f'No {fp} session found for {gp} {year}')
                fp_sessions.append(pd.DataFrame())

        #Combining list of DataFrames into one DataFrame
        fp_sessions = pd.concat(fp_sessions, ignore_index=True)
        return fp_sessions


    def get_fastest_race_lap(self, year, gp):
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
        #Collecting and loading Race session
        try:
            race_session = f1.get_session(year = year, gp = gp, identifier = 'Race')
            race_session.load()
        except:
            print(f'Could not load race session for {gp} {year}')
            return np.nan

        try:
            #Find the fastest lap of the race
            df_race_fastest_lap = race_session.laps.pick_fastest()
        except:
            return np.nan

        #Convert laptime from DateTime to float (seconds)
        #Races like the Belgian GP in 2021 returns None,
        #since every lap of the race was completed under
        #a safetycar, and therefore has no fastest lap.
        try:
            race_fastest_lap = df_race_fastest_lap['LapTime'].total_seconds()
            return race_fastest_lap
        except:
            return np.nan

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
        fp_sessions = self.get_practice_laps(year, gp)

        #Empty dataframe with columns for driver and their fastest lap
        valid_driver_fastest_lap = pd.DataFrame(columns=['DriverNumber',
                                                         'Team',
                                                         'FastestFPLap',
                                                         'MeanFPLaps',
                                                         'StdFPLaps',
                                                         'GP'])

        #If there if no FP session
        if fp_sessions.empty:
            return valid_driver_fastest_lap

        #Collect all drivers that have completed a lap in FP
        drivers = fp_sessions['DriverNumber'].unique()

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

            #Used for calculating mean later
            driver_laps_sum = 0
            driver_number_of_laps = len(valid_driver_laps)

            #Converting drivers laps to seconds
            if driver_number_of_laps != 0:
                for j in range(driver_number_of_laps):
                    driver_lap_times_sec.append(
                        valid_driver_laps.iloc[j]['LapTime'].total_seconds()
                        )
                    driver_laps_sum += driver_lap_times_sec[j]
            #If driver has not completed a lap in FP laptime is set to NaN
            else:
                driver_lap_times_sec.append(np.nan)

            #Sort drivers lap times. Fastest lap is at [0]
            driver_lap_times_sec.sort()
            driver_fastest_lap = driver_lap_times_sec[0]

            #Find average lap time of all FP sessions
            driver_lap_mean = driver_laps_sum / driver_number_of_laps

            #Calculate std
            error_squared = 0

            if driver_number_of_laps == 0:
                driver_lap_std = np.nan
            else:
                for lap in (driver_lap_times_sec):
                    error_squared += (lap - driver_lap_mean)**2
                driver_lap_std = np.sqrt(error_squared / driver_number_of_laps)

            #Add drivers fastest lap to dataframe
            valid_driver_fastest_lap.loc[len(valid_driver_fastest_lap)] = [driver,
                                                                           driver_team,
                                                                           driver_fastest_lap,
                                                                           driver_lap_mean,
                                                                           driver_lap_std,
                                                                           gp]

        #Sort dataframe before returning
        #valid_driver_fastest_lap = valid_driver_fastest_lap.sort_values(
            #by = 'FastestLap'
            #).reset_index(drop = True)

        return valid_driver_fastest_lap

    def get_data_from_api(self, years):

        #Collecting schedule for entire season, excluding testing sessions
        for year in years:
            try:
                event_schedule = f1.get_event_schedule(year, include_testing = False)
            except:
                print(f'Could not not schedule for year {year}')
                continue

            #Getting a list of all gps (['Italian Grand Prix', ...])
            gps = event_schedule['EventName'].tolist()

            #Empty list to collect all the DataFrames
            list_of_data = []

            #Adding fastest FP and Race laps and year gp took place
            for gp in gps:
                df_fp_data = self.get_fastest_laps(year, gp)
                df_fp_data['FastestLapRace'] = self.get_fastest_race_lap(year, gp)
                df_fp_data['Year'] = year
                list_of_data.append(df_fp_data)

        #Combining list of DataFrames into one DataFrame
        data = pd.concat(list_of_data, ignore_index=True)

        #Removes NaN values for set for drivers who has not completed a FP lap
        data = data.dropna(subset=['FastestLap'])

        #Adds fasten than teammate column
        data = self.faster_then_teammate(data)

        #Convert DataFrame to csv file
        data.to_csv('F1_data.csv', index=False)

    def faster_then_teammate(self, data):
        '''
        Adds 'FasterThenTeammate' column to dataframe
        '''
        data['FasterThanTeammate'] = (
            (data['FastestLap'] == data.groupby(['Year','GP','Session','Team'])['FastestLap']
            .transform('min'))
            .astype(float)
            )

         #Set NaN where only one teammate had a valid lap
        only_one_driver = data.groupby(['Year','GP','Session','Team'])['DriverNumber'].transform('count') == 1
        data.loc[only_one_driver, 'FasterThanTeammate'] = np.nan

        return data

if __name__ == '__main__':
    years = [2024, 2023, 2022]
    obj = NameTBD()
    obj.get_data_from_api(years=years)


