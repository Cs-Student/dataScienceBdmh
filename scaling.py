# -*- coding: utf-8 -*-
"""
Created on Thu May  7 06:30:54 2020

@author: shubham
"""
#Feed the data as MxN matrix where M is the number of samples and N is the number of features

import pandas as pd
import numpy as np
import copy

def MinMaxScalingTrain(data):
    temp_data = copy.deepcopy(data)
    temp_data = temp_data.T
    min_array = np.zeros(data.shape[1])
    max_array = np.zeros(data.shape[1])
    for column in range(len(temp_data)):
        mini = np.min(temp_data[column])
        maxi = np.max(temp_data[column])
        min_array[column] = mini
        max_array[column] = maxi
        if(mini != maxi):
            temp_data[column] = (temp_data[column] - mini)/(maxi - mini)
        else:
            temp_data[column] = (temp_data[column] - mini)
    temp_data = temp_data.T
    return temp_data,min_array,max_array

def MinMaxScalingTest(data,min_array,max_array):
    temp_data = copy.deepcopy(data)
    temp_data = temp_data.T
    for column in range(len(temp_data)):
        mini = min_array[column]
        maxi = max_array[column]
        if(mini != maxi):
            temp_data[column] = (temp_data[column] - mini)/(maxi - mini)
        else:
            temp_data[column] = (temp_data[column] - mini)
    temp_data = temp_data.T
    return temp_data

def ZScoreScalingTrain(data):
    temp_data = copy.deepcopy(data)
    temp_data = temp_data.T
    mean_array = np.zeros(data.shape[1])
    std_array = np.zeros(data.shape[1])
    for column in range(len(temp_data)):
        mean = np.mean(temp_data[column])
        std = np.std(temp_data[column])
        mean_array[column] = mean
        std_array[column] = std
        if(std != 0):
            temp_data[column] = (temp_data[column] - mean)/std   
        else:
            temp_data[column] = (temp_data[column] - mean)
    temp_data = temp_data.T
    return temp_data,mean_array,std_array

def ZScoreScalingTest(data,mean_array,std_array):
    temp_data = copy.deepcopy(data)
    temp_data = temp_data.T
    for column in range(len(temp_data)):
        mean = mean_array[column]
        std = std_array[column]
        if(std != 0):
            temp_data[column] = (temp_data[column] - mean)/std   
        else:
            temp_data[column] = (temp_data[column] - mean)    
    temp_data = temp_data.T
    return temp_data


