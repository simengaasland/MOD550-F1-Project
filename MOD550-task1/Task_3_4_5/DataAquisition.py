import fastf1 as f1
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

class DataAquisition: 
    '''
    A class with two functions that...
    '''
    
    def __init__(self, gp, year): 
        '''
        Input
        ------
        gp: string 
            The Grand Prix to collect data from (e.g. 'Monza').

        year: int 
            The year the Grand Prix took place (e.g. 2025).
        '''
        self.gp = gp
        self.year = year

    def get_practice_laps(self): 
        '''
        Collects FP1, FP2 and FP3 data from 
        the FastF1 API. 

        Output
        ------
        DataFrame
            One DataFrame for each session. 
        '''
        FP1_session = f1.get_session(year = self.year, gp = self.gp, identifier = 'FP1')
        FP2_session = f1.get_session(year = self.year, gp = self.gp, identifier = 'FP2')
        FP3_session = f1.get_session(year = self.year, gp = self.gp, identifier = 'FP3')

        FP1_session.load()
        FP2_session.load()
        FP3_session.load()

        df_FP1_session = FP1_session.laps
        df_FP2_session = FP2_session.laps
        df_FP3_session = FP3_session.laps

        return df_FP1_session, df_FP2_session, df_FP3_session

    def get_fastest_laps(self): 
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
            The Grand Prix to collect data from (e.g. 'Monza').

        year: int 
            The year the Grand Prix took place (e.g. 2025).

        Output
        ------
        DataFrame
            A DataFrame with driver number, session and lap 
            time in seconds 
        '''
        list_of_FP_sessions = self.get_practice_laps()
        
        #Empty dataframe with columns for driver and their fastest lap
        valid_driver_fastest_lap = pd.DataFrame(columns=['DriverNumber','Session','FastestLap'])
        
        for session in range(len(list_of_FP_sessions)):
            #Collect all drivers that have completed a lap in FP 
            drivers = list_of_FP_sessions[session]['DriverNumber'].unique()
        
            for i in range(len(drivers)):
                #Collect all laps completed by driver
                driver_laps = list_of_FP_sessions[session][list_of_FP_sessions[session]['DriverNumber'] == drivers[i]]
                
                #Remove deleted laps and NaN lap times 
                valid_driver_laps = driver_laps[driver_laps['Deleted'].astype(bool) == False].dropna(subset=['LapTime'])
    
                #Temp list to store laps in total seconds 
                driver_lap_times_sec = []
    
                #Converting drivers laps to seconds
                for j in range(len(valid_driver_laps)):
                    driver_lap_times_sec.append(valid_driver_laps.iloc[j]['LapTime'].total_seconds())
    
                #Sort drivers lap times. Fastest lap is at [0]
                driver_lap_times_sec.sort()
            
                #Add drivers fastest lap to dataframe
                valid_driver_fastest_lap.loc[len(valid_driver_fastest_lap)] = [drivers[i],f'FP{session + 1}' , driver_lap_times_sec[0]]

        #Sort dataframe before returning 
        valid_driver_fastest_lap = valid_driver_fastest_lap.sort_values(by = 'FastestLap').reset_index(drop = True)    
        
        return valid_driver_fastest_lap

    def plot_histogram(self, bins = 12):
            '''
            Plots a histogram of the lastest laps

            Input
            ------
            bins: int
                Amount of bins in the histogram plot
            '''
            df = self.get_fastest_laps()
            
            plt.hist(df['FastestLap'], bins = bins, color = 'green', edgecolor = 'black')
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency")
            plt.title(f"Fastest laps {self.gp} GP {self.year} FP1/FP2/FP3")
            plt.show()

    def plot_pmf(self, bins):
        '''
            Plots the Probability Mass Function (PMF)

            Input
            ------
            bins: int
                Amount of bins in the PMF plot
            '''
        df = self.get_fastest_laps()
        frequency_count, bin_edges = np.histogram(df['FastestLap'], bins = bins)

        #Calc probability of each bin
        pmf = frequency_count / frequency_count.sum() 

        #Plot PMF
        plt.stem(bin_edges[:-1], pmf)
        plt.title(f"Discrete PMF fastest laps {self.gp} GP {self.year} FP1/FP2/FP3")
        plt.xlabel("Time [s]")
        plt.ylabel("Probability")
        plt.show()

    def plot_cumulative_pmf(self, bins):
        '''
            Plots the cumulative Probability Mass Function (PMF)

            Input
            ------
            bins: int
                Amount of bins in the PMF plot
            '''

        df = self.get_fastest_laps()
        frequency_count, bin_edges = np.histogram(df['FastestLap'], bins = bins)

        #Calc probability of each bin
        pmf = frequency_count / frequency_count.sum() 
        
        cumulative_pmf = []
        cumulative_value = 0

        #Calculate cumulative value for each bin
        for i in range(len(pmf)):
            cumulative_value += pmf[i]
            cumulative_pmf.append(cumulative_value)
        
        plt.stem(bin_edges[:-1], cumulative_pmf)
        plt.title(f"Cumulative Discrete PMF fastest laps {self.gp} GP {self.year} FP1/FP2/FP3")
        plt.xlabel("Time [s]")
        plt.ylabel("Probability")
        plt.show()