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
print("**Univariate analysis of F1 dataset**")
print("Mean: ", mean(Dataset_F1['Average Car Weight (kg)']))
print("Mean (using NumPy): ", np.mean(Dataset_F1['Average Car Weight (kg)']))
print("Variance: ", variance(Dataset_F1['Average Car Weight (kg)']))
print("Variance (using NumPy): ", np.var(Dataset_F1['Average Car Weight (kg)']))
print("Standard Deviation: ", std_dev(Dataset_F1['Average Car Weight (kg)']))
print("Standard Deviation (using NumPy): ", np.std(Dataset_F1['Average Car Weight (kg)']))
plot.hist(Dataset_F1['Average Car Weight (kg)'], bins = 5, color = 'blue', edgecolor = 'black')
plot.xlabel('Average Car Weight (kg)')
plot.ylabel('Frequency')
plot.title('Histogram of Average Car Weight (kg) in F1 dataset')
plot.grid(True)
plot.show()

print("**Univariate analysis of IMDB dataset**")
print("Mean: ", mean(Dataset_IMDB['numVotes']))
print("Mean (using NumPy): ", np.mean(Dataset_IMDB['numVotes']))
print("Variance: ", variance(Dataset_IMDB['numVotes']))
print("Variance (using NumPy): ", np.var(Dataset_IMDB['numVotes']))
print("Standard Deviation: ", std_dev(Dataset_IMDB['numVotes']))
print("Standard Deviation (using NumPy): ", np.std(Dataset_IMDB['numVotes']))
plot.hist(Dataset_IMDB['numVotes'], bins = 30, color = 'blue', edgecolor = 'black')
plot.xlabel('numVotes')
plot.ylabel('Frequency')
plot.title('Histogram of Average Rating in IMDB dataset')
plot.grid(True)
plot.show()

print("**Univariate analysis of NVDA dataset**")
print("Mean: ", mean(Dataset_NVDA['Close']))
print("Mean (using NumPy): ", np.mean(Dataset_NVDA['Close']))
print("Variance: ", variance(Dataset_NVDA['Close']))
print("Variance (using NumPy): ", np.var(Dataset_NVDA['Close']))
print("Standard Deviation: ", std_dev(Dataset_NVDA['Close']))
print("Standard Deviation (using NumPy): ", np.std(Dataset_NVDA['Close']))
plot.hist(Dataset_NVDA['Close'], bins = 30, color = 'blue', edgecolor = 'black')
plot.xlabel('Close')
plot.ylabel('Frequency')
plot.title('Histogram of Close in NVDA dataset')
plot.grid(True)
plot.show()


def covariance(data_entries1, data_entries2):
    n = 0
    sum1 = 0
    sum2 = 0
    sum1_sq = 0
    sum2_sq = 0
    sum_product = 0
    for entry1, entry2 in zip(data_entries1, data_entries2):
        sum1 += entry1
        sum2 += entry2
        sum1_sq += entry1**2
        sum2_sq += entry2**2
        sum_product += entry1*entry2
        n += 1
    
    return (sum_product/n) - (sum1/n)*(sum2/n)

# ------------------------------------                       ------------------------------------ #
# ------------------------------------  JOINT DISTRIBUTION   ------------------------------------ #
# ------------------------------------                       ------------------------------------ #
print("**Joint distribution of F1 dataset (Weight vs Overtakes)**")
cov = covariance(Dataset_F1['Average Car Weight (kg)'], Dataset_F1['Overtakes'])
print("Covariance: ", cov)
print("Covariance (using NumPy): ", np.cov(Dataset_F1['Average Car Weight (kg)'], Dataset_F1['Overtakes'])[0][1])
correlation = cov/(std_dev(Dataset_F1['Average Car Weight (kg)'])*std_dev(Dataset_F1['Overtakes']))
print("Correlation: ", correlation)
print("Correlation (using NumPy): ", np.corrcoef(Dataset_F1['Average Car Weight (kg)'], Dataset_F1['Overtakes'])[0][1])
plot.scatter(Dataset_F1['Average Car Weight (kg)'], Dataset_F1['Overtakes'], color = 'blue')
plot.xlabel('Average Car Weight (kg)')
plot.ylabel('Overtakes')
plot.title('Scatter plot of Average Car Weight (kg) vs Overtakes in F1 dataset')
# linear regression
m, b = np.polyfit(Dataset_F1['Average Car Weight (kg)'], Dataset_F1['Overtakes'], 1)
predicted_overtakes = m * Dataset_F1['Average Car Weight (kg)'] + b
plot.plot(Dataset_F1['Average Car Weight (kg)'], predicted_overtakes, color='red', label='Regression Line')
plot.grid(True)
plot.tight_layout()
plot.show()

print("**Joint distribution of IMDB dataset (Number of Votes vs Year of Release)**")
cov = covariance(Dataset_IMDB['numVotes'], Dataset_IMDB['releaseYear'])
print("Covariance: ", cov)
print("Covariance (using NumPy): ", np.cov(Dataset_IMDB['numVotes'], Dataset_IMDB['releaseYear'])[0][1])
correlation = cov/(std_dev(Dataset_IMDB['numVotes'])*std_dev(Dataset_IMDB['releaseYear']))
print("Correlation: ", correlation)
print("Correlation (using NumPy): ", np.corrcoef(Dataset_IMDB['numVotes'], Dataset_IMDB['releaseYear'])[0][1])
plot.scatter(Dataset_IMDB['numVotes'], Dataset_IMDB['releaseYear'], color = 'blue')
plot.xlabel('numVotes')
plot.ylabel('releaseYear')
plot.title('Scatter plot of Number of Votes vs Release Year in IMDB dataset')
# linear regression
m, b = np.polyfit(Dataset_IMDB['numVotes'], Dataset_IMDB['releaseYear'], 1)
predicted_votes = m * Dataset_IMDB['numVotes'] + b
plot.plot(Dataset_IMDB['numVotes'], predicted_votes, color='red', label='Regression Line')
plot.grid(True)
plot.tight_layout()
plot.show()

print("**Joint distribution of NVDA dataset (Close vs Volume)**")
cov = covariance(Dataset_NVDA['Close'], Dataset_NVDA['Volume'])
print("Covariance: ", cov)
print("Covariance (using NumPy): ", np.cov(Dataset_NVDA['Close'], Dataset_NVDA['Volume'])[0][1])
correlation = cov/(std_dev(Dataset_NVDA['Close'])*std_dev(Dataset_NVDA['Volume']))
print("Correlation: ", correlation)
print("Correlation (using NumPy): ", np.corrcoef(Dataset_NVDA['Close'], Dataset_NVDA['Volume'])[0][1])
plot.scatter(Dataset_NVDA['Close'], Dataset_NVDA['Volume'], color = 'blue')
plot.xlabel('Close')
plot.ylabel('Volume')
plot.title('Scatter plot of Close vs Volume in NVDA dataset')
# linear regression
m, b = np.polyfit(Dataset_NVDA['Close'], Dataset_NVDA['Volume'], 1)
predicted_volume = m * Dataset_NVDA['Close'] + b
plot.plot(Dataset_NVDA['Close'], predicted_volume, color='red', label='Regression Line')
plot.grid(True)
plot.tight_layout()
plot.show()