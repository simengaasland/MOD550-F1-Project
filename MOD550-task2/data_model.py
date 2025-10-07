'''
This module contains the DataModel class
'''

import sys
import random

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


sys.path.append('C:/Users/simen/Documents/GitHub/MOD550-F1-Project/MOD550-task1')
from data_generator import DataGenerator

class DataModel:
    '''
    Fitting docstring for this class
    '''
    def __init__(self,n_points):
        self.n = n_points
        self.dg = DataGenerator(self.n)
        self.data = self.dg.random_2d_point_gen()

    def simple_linear_regression(self, statistic = False):
        '''
        Doc string
        '''
        if statistic is True:
            train_data = self.data
        else:
            train_data, self.valid_data, _ = self.split_train_validation_test()

        #Splits up data into x- and y-values
        x_train, y_train = self.dg.get_xy_values(train_data)

        #Calculates mean of x- and y-values
        sum_x = 0
        sum_y = 0
        mean_x = 0
        mean_y = 0

        for _, x_val in enumerate(x_train):
            sum_x += x_val
        mean_x = sum_x / self.n

        for _, y_val in enumerate(y_train):
            sum_y += y_val
        mean_y = sum_y / self.n

        #Calculate slope of regression line
        nominator_sum = 0
        denominator_sum = 0

        for i in range(len(train_data)):
            nominator_sum += (x_train[i] - mean_x) * (y_train[i] - mean_y)
            denominator_sum += (x_train[i] - mean_x)**2

        self.m = nominator_sum / denominator_sum

        #Calculate intercept y = m*x + c
        self.c = mean_y - (self.m * mean_x)

        #Calculate y = m*x + c
        self.y_pred = []
        for _, x in enumerate(x_train):
            self.y_pred.append(self.m*x + self.c)

        #Plot
        plt.scatter(x_train, y_train, color = 'blue', label = 'Training data')
        plt.plot(x_train, self.y_pred, color = 'red', label = 'Prediction')
        plt.title(r'$\hat y = {round(self.m,2)} \cdot x + {round(self.c,2)}$')
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

    def calc_mse(self):
        '''
        This function uses validation data to calulate Mean squared Error.
        If a linear regression has not been preformed the fuction will
        return mse = None.
        '''
        try:
            #Get x- and y-values
            x_valid, y_valid = self.dg.get_xy_values(self.valid_data)

            #Initialize
            mse = 0
            error_squared = 0

            #Calculate error squared
            for i, y_real in enumerate(y_valid):
                y_pred = self.m * x_valid[i] + self.c
                error_squared += (y_real - y_pred)**2

            #Calculate Mean Errro Squared and return
            mse = (1/self.n) * error_squared
            return round(mse,2)

        except Exception as e:
            print('A linear regression has to be preformed before using calc_MSE function')
            print(f'Error: {e}')
            mse = None

    def neural_network(self):
        '''
        Doc string
        '''
        #Gets training data
        train_data, _ , test_data = self.split_train_validation_test()

        #Splits up data into x- and y-values
        x_train, y_train = self.dg.get_xy_values(train_data)
        x_test, y_test = self.dg.get_xy_values(test_data)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        model = Sequential([
            Dense(32, input_dim = 1, activation = 'relu', kernel_regularizer =l2(0.01)),
            Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
            Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
            Dense(1)])

        model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

        model.fit(x_train,y_train, epochs = 70, verbose = 1)

        y_pred = model.predict(x_test)
        print(x_train.shape)
        plt.scatter(x_train, y_train, color = 'blue', label = 'Training data')
        plt.plot(x_test, y_pred, color = 'red', label = 'NN prediction')
        plt.legend()
        plt.show()

    def k_mean(self, n_clusters, elbow_plot = True):
        '''
        Doc string
        '''
        x, y = self.dg.get_xy_values(self.data)

        if elbow_plot is True:
            inertias = []

            for i in range(1,11):
                kmeans = KMeans(n_clusters=i)
                kmeans.fit(self.data)
                inertias.append(kmeans.inertia_)

            plt.plot(range(1,11), inertias, marker='x')
            plt.title('Elbow method')
            plt.xlabel('Number of clusters')
            plt.ylabel('Within SS')
            plt.show()

        model = KMeans(n_clusters = n_clusters)
        model.fit(self.data)

        plt.scatter(x, y, c = model.labels_)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'K means with {n_clusters} clusters')
        plt.show()

    def gmm(self, n_clusters):
        '''
        Doc string
        '''
        x, y = self.dg.get_xy_values(self.data)

        model = GaussianMixture(n_components = n_clusters)
        model.fit(self.data)
        labels = model.predict(self.data)

        plt.scatter(x, y, c = labels)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Gaussian Mixture Model (GMM) with {n_clusters} clusters')
        plt.show()

