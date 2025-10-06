'''
This module contains the DataModel class
'''
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append('C:/Users/simen/Documents/GitHub/MOD550-F1-Project/MOD550-task1')
from DataGenerator import DataGenerator

class DataModel:
    '''
    Fitting docstring for this class
    '''
    def __init__(self,n_points):
        self.N = n_points
        self.dg = DataGenerator(self.N)
        self.data = self.dg.random_2d_point_gen()


    def simple_linear_regression(self):

        #Splits up data into x- and y-values
        data_x_values, data_y_values = self.dg.get_xy_values(self.data)

        #Calculates mean of x- and y-values
        sum_x = 0
        sum_y = 0
        mean_x = 0
        mean_y = 0

        for _, x_val in enumerate(data_x_values):
            sum_x += x_val
        mean_x = sum_x / self.N

        for _, y_val in enumerate(data_y_values):
            sum_y += y_val
        mean_y = sum_y / self.N

        #Calculate slope of regression line
        nominator_sum = 0
        denominator_sum = 0

        for i in range(self.N):
            nominator_sum += (data_x_values[i] - mean_x) * (data_y_values[i] - mean_y)
            denominator_sum += (data_x_values[i] - mean_x)**2

        m = nominator_sum / denominator_sum

        #Calculate intercept y = m*x + c
        c = mean_y - (m * mean_x)

        #Calculate y = m*x + c
        y_pred = []
        for _, x in enumerate(data_x_values):
            y_pred.append(m*x + c)

        #Plot
        plt.scatter(data_x_values, data_y_values, color = 'blue')
        plt.plot(data_x_values, y_pred, color = 'red')
        plt.title(f'$ \hat y = {round(m,2)} \cdot x + {round(c,2)}$')
        plt.show()

    def  split_train_validation_test(self):
        '''
        This function randomly shuffles the data
        and splits it into train, validation and
        test. Split is set as follows:
        Train : 70%
        Validation: 10%
        Test: 20%
        '''
        #Shuffling the data randomly
        random.shuffle(self.data)

        #Finding total data point (could just use self.N)
        len_data = len(self.data)

        #Validation data is 10% of total data
        valid_item = int(round(len_data * 0.1))

        #Validation is 20% of total data
        test_item = valid_item + int(round(len_data * 0.2))

        #Rest is train data (70%)

        #Splitting the data
        valid_data = self.data[0 : valid_item]
        test_data = self.data[valid_item : test_item]
        train_data = self.data[test_item : ]

        return train_data, valid_data, test_data




class1 = DataModel(n_points = 100)
#class1.simple_linear_regression()
#train_data, valid_data, test_data = class1.split_train_validation_test()
