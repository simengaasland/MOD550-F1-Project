'''
This module contains the DataModel class
'''
import sys
import pandas as pd
import numpy as np
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


class1 = DataModel(n_points = 100)
class1.simple_linear_regression()