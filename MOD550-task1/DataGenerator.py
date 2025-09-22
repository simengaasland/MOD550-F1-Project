import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import seaborn as sb

class DataGenerator: 
    def __init__(self, n_points):
        self.n_points = n_points

    def random_2d_point_gen(self):
        '''
        Generates a list of random points in the 
        xy plane. The x- and y-values range from 
        0 - 10. 

        Input
        -------
        n_points: int 
            Amount of points (x, y) 

        Returns
        -------
        list
            List of points (x, y)
        '''
        
        rnd_list = []

        #Creates list with random points
        for i in range(self.n_points):
            x = round(rnd.random()*10, 2)
            y = round(rnd.random()*10, 2)
            rnd_list.append([x,y])

        return rnd_list
        
    def get_xy_values(self, point_list):
        '''
        Takes a list of point and generates 
        two lists, one with x-values and the
        other one with y-values.
        
        Input
        ------
        point_list: list 
            A list of points in the xy plane
            
        Output
        ------
        list 
            List of x-values 
        list 
            List of y-values 
        '''
        x_values = []
        y_values = [] 

        #Gets all the x- and y-values and puts them i seperate lists
        for i in range(len(point_list)):
            x_values.append(point_list[i][0])
            y_values.append(point_list[i][1])

        return x_values, y_values

    def plot_histogram(self, point_list, bins):
        '''
        Plots a histogram from points

        Input
        -----
        point_list: list 
            A list of points in the xy plane
        bins: int
            Amount of bins in the histogram plot
        '''
        x_values, y_values = self.get_xy_values(point_list)
        
        #Plot histogram for x-values
        plt.hist(x_values, bins = bins, color='blue', edgecolor='black')
        plt.xlabel("x-values")
        plt.ylabel("Frequency")
        plt.title("Histogram of x-values")
        plt.show()

        #Plot histogram for y-values
        plt.hist(y_values, bins = bins, color='red', edgecolor='black')
        plt.xlabel("y-values")
        plt.ylabel("Frequency")
        plt.title("Histogram of y-values")
        plt.show()

    def plot_heatmap(self, point_list):
        '''
        Plots a heatmap from points

        Input
        -----
        point_list: list 
            A list of points in the xy plane
        '''
        x_values, y_values = self.get_xy_values(point_list)
        
        heatmap,*_ = np.histogram2d(x_values, y_values, bins=10)

        plt.imshow(heatmap.T, origin='lower', cmap = 'hot')
        plt.colorbar(label='Density')
        plt.title('Heatmap')
        plt.xlabel('x-value')
        plt.ylabel('y-value')
        plt.show()
