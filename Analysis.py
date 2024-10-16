# importing the required libraries
import pandas as pd # to read dataset
import numpy as np # for calculations
import matplotlib.pyplot as plot # for plotting

Dataset_F1 = pd.read_csv('./f1.csv')
Dataset_IMDB = pd.read_csv('./imdb.csv')
Dataset_NVDA = pd.read_csv('./NVDA.csv')

# ------------------------------------                       ------------------------------------ #
# ------------------------------------  UNIVARIATE ANALYSIS  ------------------------------------ #
# ------------------------------------                       ------------------------------------ #

# Function definitions for univariate analysis
def mean(data_entries):
    n = 0
    sum = 0
    for entry in data_entries:
        sum += entry
        n += 1
    
    return sum/n

def variance(data_entries):
    n = 0
    sum = 0
    sum_sq = 0
    for entry in data_entries:
        sum += entry
        sum_sq += entry**2
        n += 1
    
    return (sum_sq/n) - (sum/n)**2

def std_dev(data_entries):
    return variance(data_entries)**0.5

# Displaying univariate analysis
print("Univariate analysis of F1 dataset")
print("Mean: ", mean(Dataset_F1['Average Car Weight (kg)']))
print("Variance: ", variance(Dataset_F1['Average Car Weight (kg)']))
print("Standard Deviation: ", std_dev(Dataset_F1['Average Car Weight (kg)']))
plot.hist(Dataset_F1['Average Car Weight (kg)'], bins = 20, color = 'blue', edgecolor = 'black')
plot.xlabel('Average Car Weight (kg)')
plot.ylabel('Frequency')
plot.title('Histogram of Average Car Weight (kg) in F1 dataset')
plot.show()

print("Univariate analysis of IMDB dataset")
print("Mean: ", mean(Dataset_IMDB['averageRating']))
print("Variance: ", variance(Dataset_IMDB['averageRating']))
print("Standard Deviation: ", std_dev(Dataset_IMDB['averageRating']))
plot.hist(Dataset_IMDB['averageRating'], bins = 20, color = 'blue', edgecolor = 'black')
plot.xlabel('averageRating')
plot.ylabel('Frequency')
plot.title('Histogram of Average Rating in IMDB dataset')
plot.show()

print("Univariate analysis of NVDA dataset")
print("Mean: ", mean(Dataset_NVDA['Close']))
print("Variance: ", variance(Dataset_NVDA['Close']))
print("Standard Deviation: ", std_dev(Dataset_NVDA['Close']))
plot.hist(Dataset_NVDA['Close'], bins = 20, color = 'blue', edgecolor = 'black')
plot.xlabel('Close')
plot.ylabel('Frequency')
plot.title('Histogram of Close in NVDA dataset')
plot.show()

