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

#--------------------------------------Task 2.1--------------------------------------
sys.path.append('C:/Users/simen/Documents/GitHub/MOD550-F1-Project/MOD550-task1')
from data_generator import DataGenerator

class DataModel:

    def __init__(self,n_points):
        self.n = n_points
        self.dg = DataGenerator(self.n)
        self.data = self.dg.random_2d_point_gen()

#--------------------------------------Task 2.2--------------------------------------
    def simple_linear_regression(self, statistic = False):
        '''
        This function preforms a simple linear regression
        and plots the result.
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
        plt.title(f'$\hat y = {self.m : .2f} \cdot x + {self.c : .2f}$')
        plt.show()

#--------------------------------------Task 2.3--------------------------------------
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

#--------------------------------------Task 2.4--------------------------------------
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
            return mse

        except Exception as e:
            print('A linear regression has to be preformed (statistic = False) ' \
            'before using calc_mse function')
            print(f'Error: {e}')
            mse = None

#--------------------------------------Task 2.5--------------------------------------
    def neural_network(self):
        '''
        This functions sets up a Neural Network (NN) with
        3 hidden layers and 1 output layer. The result
        is then plotted.
        '''
        #Gets training data
        train_data, _ , test_data = self.split_train_validation_test()

        #Splits up data into x- and y-values
        x_train, y_train = self.dg.get_xy_values(train_data)
        x_test, y_test = self.dg.get_xy_values(test_data)

        #Convert to arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        #Build NN with 3 hidden layers and 1 output layer
        model = Sequential([
            Dense(32, input_dim = 1, activation = 'relu', kernel_regularizer =l2(0.01)),
            Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
            Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
            Dense(1)])

        #Fitting model
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
        model.fit(x_train,y_train, epochs = 70, verbose = 0)

        #Make prediction
        y_pred = model.predict(x_test)

        #Plot
        plt.scatter(x_train, y_train, color = 'blue', label = 'Training data')
        plt.plot(x_test, y_pred, color = 'red', label = 'NN prediction')
        plt.legend()
        plt.show()

#--------------------------------------Task 2.6--------------------------------------
    def k_mean(self, n_clusters, elbow_plot = True):
        '''
        This function uses k_means to divide unlabeled data
        into clusters. The number of clusters is defined by
        n_clusters. The elbow plot can be used to choose
        number of clusters.
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
        This function uses a Gaussian Mixture Model (GMM)
        to devide unlabeled data into clusters. The amount
        of clusters if defined by n_clusters.
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

