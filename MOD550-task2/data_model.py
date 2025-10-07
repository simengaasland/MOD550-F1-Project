'''
This module contains the DataModel class
'''
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score

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

        train_data, self.valid_data, _ = self.split_train_validation_test()

        #Splits up data into x- and y-values
        X_train, y_train = self.dg.get_xy_values(train_data)

        #Calculates mean of x- and y-values
        sum_x = 0
        sum_y = 0
        mean_x = 0
        mean_y = 0

        for _, x_val in enumerate(X_train):
            sum_x += x_val
        mean_x = sum_x / self.N

        for _, y_val in enumerate(y_train):
            sum_y += y_val
        mean_y = sum_y / self.N

        #Calculate slope of regression line
        nominator_sum = 0
        denominator_sum = 0

        for i in range(len(train_data)):
            nominator_sum += (X_train[i] - mean_x) * (y_train[i] - mean_y)
            denominator_sum += (X_train[i] - mean_x)**2

        self.m = nominator_sum / denominator_sum

        #Calculate intercept y = m*x + c
        self.c = mean_y - (self.m * mean_x)

        #Calculate y = m*x + c
        self.y_pred = []
        for _, x in enumerate(X_train):
            self.y_pred.append(self.m*x + self.c)

        #Plot
        plt.scatter(X_train, y_train, color = 'blue', label = 'Training data')
        plt.plot(X_train, self.y_pred, color = 'red', label = 'Prediction')
        plt.title(f'$ \hat y = {round(self.m,2)} \cdot x + {round(self.c,2)}$')
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

    def calc_MSE(self):
        '''
        This function uses validation data to calulate Mean squared Error.
        If a linear regression has not been preformed the fuction will
        return mse = None.
        '''
        try:
            #Get x- and y-values
            valid_x_values, valid_y_values = self.dg.get_xy_values(self.valid_data)

            #Initialize
            mse = 0
            error_squared = 0

            #Calculate error squared
            for i, y_real in enumerate(valid_y_values):
                y_pred = self.m * valid_x_values[i] + self.c
                error_squared += (y_real - y_pred)**2

            #Calculate Mean Errro Squared and return
            mse = (1/self.N) * error_squared
            return round(mse,2)

        except:
            print('A linear regression has to be preformed before using calc_MSE function')
            mse = None

    def neural_network(self):
        '''
        Doc string
        '''
        #Gets training data
        train_data, _ , test_data = self.split_train_validation_test()

        #Splits up data into x- and y-values
        X_train, y_train = self.dg.get_xy_values(train_data)
        X_test, y_test = self.dg.get_xy_values(test_data)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        model = Sequential([Dense(32, input_dim = 1, activation = 'relu', kernel_regularizer =l2(0.01)),
                            Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
                            Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
                            Dense(1)])

        model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

        model.fit(X_train,y_train, epochs = 70, verbose = 1)

        y_pred = model.predict(X_test)
        print(X_train.shape)
        plt.scatter(X_train, y_train, color = 'blue', label = 'Training data')
        plt.plot(X_test, y_pred, color = 'red', label = 'NN prediction')
        plt.legend()
        plt.show()

    def K_mean(self):




class1 = DataModel(n_points = 100)
#class1.simple_linear_regression()
#train_data, valid_data, test_data = class1.split_train_validation_test()
#print(class1.calc_MSE())
class1.neural_network()

