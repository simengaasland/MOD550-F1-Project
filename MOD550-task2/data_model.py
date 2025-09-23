'''
This module contains the DataModel class
'''
import sys
import pandas as pd
import fastf1 as f1

sys.path.append('C:/Users/simen/Documents/GitHub/MOD550-F1-Project/MOD550-task1')
from data_aquisition import DataAquisition

class DataModel:
    '''
    Fitting docstring for this class
    '''
    def __init__(self):
        year = 2024
        #Collecting schedule for entire season, excluding testing sessions
        event_schedule = f1.get_event_schedule(year, include_testing = False)

        #Removing sprint weekends as they do not have FP2 and FP3
        conventinal_event_schedule = event_schedule[
            event_schedule['EventFormat'] == 'conventional'
            ]

        #Getting a list of all gps (['Sakhir', 'Jeddah' ...])
        gps = conventinal_event_schedule['Location'].tolist()

        #Empty list to collect all the DataFrames
        list_of_data = []

        for _, gp in enumerate(gps):
            da = DataAquisition(gp, year)
            df_fp_data = da.get_fastest_laps()
            df_fp_data['FastestLapRace'] = da.get_fastest_race_lap()
            list_of_data.append(df_fp_data)

        #Combining list of DataFrames into one DataFrame
        self.data = pd.concat(list_of_data, ignore_index=True)

        #Removes NaN values for set for drivers who has not completed a FP lap
        self.data = self.data.dropna(subset=['FastestLap'])
        self.data.to_csv('F1_data.csv', index=False)

    def show_data(self):
        print(self.data)


cake = DataModel()
print(cake.show_data())