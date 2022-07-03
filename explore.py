import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import pylab
import statsmodels.api as sm

plt.rcParams["figure.figsize"] = (14, 8)

import wrangle

def train_test_split(aquifer):
    '''
    This function takes in a prepared and cleaned aquifer dataframe and splits it in preparation for EDA. The test size is 15% of the total data. 
    The function also displays a plot of the distribution of the split and returns train and test dataframes.

    Parameters: aquifer - the aquifer dataframe previously pulled in and cleaned

    Returns:    train - dataframe of the train set of aquifer ready for eda and modeling
                test - dataframe of the test set of aquifer ready for modeling
    '''
    train_size = .85
    n = aquifer.shape[0]
    test_start_index = round(train_size * n)

    train = aquifer[:test_start_index] # everything up (not including) to the test_start_index
    test = aquifer[test_start_index:] # everything from the test_start_index to the end

    plt.plot(train.index, train.water_level_elevation)
    plt.plot(test.index, test.water_level_elevation)
    return train, test

def distribution_graphs(train):
	'''
	This function takes in the train dataframe and plots out a QQ (Quantile-Quantile Plot) as well as a histogram to show how well the data approximates a normal distribution 
	
	Parameters: train - dataframe containing the target variable ready for EDA
	'''

	plt.figure(figsize=(15,6))
	plt.subplot(121)
	stats.probplot(train.water_level_elevation, plot = pylab)
	plt.subplot(122)
	train.water_level_elevation.plot(kind='hist')
	plt.show()

def monthly_distribution(train):
	'''
	This function takes in the train dataframe and plots the average water level separated by month to see if there are any differences that might show up seasonally.

	Parameters: train - dataframe containing the target variable ready for EDA
	'''
	ax = train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().plot.bar(width=.9, ec='black')
	plt.xticks(rotation=0)
	ax.set(title='Average Water Level by Month', xlabel='Month', ylabel='Water Level Elevation (ft)')
	plt.axhline(train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().mean(), color='orange', label = "Mean Average of all Data")
	plt.axhline(train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().max(), color='red', label = "Highest Average month")
	plt.axhline(train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().min(), color='yellow', label = "Lowest Average month")
	plt.ylim(600,700)
	plt.legend()
	plt.show()

def change_by_month(train):
	'''
	This function takes in the train dataframe and plots change over time when looking at month to month averages.

	Parameters: train - dataframe containing the target variable ready for EDA
	'''

	y = train.water_level_elevation
	y.resample('M').mean().diff().plot(title='Average month-to-month change in water level elevation')
	plt.show()

def annual_seasonality(train):
	'''
	This function takes in the train dataframes and plots annual seasonality when observing by the decade.

	Parameters: train - dataframe containing the target variable ready for EDA
	'''
	y = train.water_level_elevation
	y.groupby([(y.index.year//10)*10, y.index.strftime('%m-%b')]).mean().unstack(0).plot(title='Seasonal Plot')
	plt.legend(loc='lower left')
	plt.show()

def lag_plots(train):
	y = train.water_level_elevation

	plt.subplot(231)
	plt.scatter(y, y.shift(-1))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 1}$')
	plt.title('Lag of 1 Day')

	plt.subplot(232)
	plt.scatter(y, y.shift(-7))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 7}$')
	plt.title('Lag of 1 week')

	plt.subplot(233)
	plt.scatter(y, y.shift(-30))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 30}$')
	plt.title('Lag of 1 month')

	plt.subplot(234)
	plt.scatter(y, y.shift(-183))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 183}$')
	plt.title('Lag of 6 months')

	plt.subplot(235)
	plt.scatter(y, y.shift(-365))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 365}$')
	plt.title('Lag of 1 year')

	plt.subplot(236)
	plt.scatter(y, y.shift(-730))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 730}$')
	plt.title('Lag of 2 Years')

def seasonal_decomposition_graphs(train):
	y = train.water_level_elevation.asfreq('W', method = 'bfill')
	result = sm.tsa.seasonal_decompose(y)
	result.plot()
	None

def precip_elevation_monthly_corr_test(train, weather):
	monthly_resample = train.resample('M').mean()
	monthly_resample['precipitation'] = weather.total_monthly_precip.resample('M').mean()

	# Running a pearson-r correlation test, printing out if the result is statistically significant and the r correlation value.
	corr, p = stats.pearsonr(monthly_resample.dropna().water_level_elevation, monthly_resample.dropna().precipitation)

	print(f'The pearson r test shows a result of {p} and an r value of {corr}.\nSee graph below.')

	sns.lmplot(data = monthly_resample.dropna(), y='water_level_elevation', x ='precipitation')

def precip_elevation_yearly_corr_test(train, weather):
	yearly_resample = train.copy()
	yearly_resample = yearly_resample.resample('A').mean()
	yearly_resample['precipitation'] = (weather.total_monthly_precip.resample('A').mean()).interpolate(method='polynomial', order=2)

	# Running a pearson-r correlation test, printing out if the result is statistically significant and the r correlation value.
	corr, p = stats.pearsonr(yearly_resample.water_level_elevation.dropna(), yearly_resample.precipitation.dropna())

	print(f'The pearson r test shows a result of {p} and an r value of {corr}.\nSee graph below.')

	sns.lmplot(data = yearly_resample, y='water_level_elevation', x ='precipitation')


def precip_usage_yearly_corr_test(train, weather, usage):
	yearly_resample = train.copy()
	yearly_resample = yearly_resample.resample('A').mean()
	yearly_resample['precipitation'] = (weather.total_monthly_precip.resample('A').mean()).interpolate(method='polynomial', order=2)
	yearly_resample['total_water_consumption'] = (usage.total_consumption.resample('A').mean()).interpolate(method='polynomial', order=2).astype('int64')

	# Running a pearson-r correlation test, printing out if the result is statistically significant and the r correlation value.
	corr, p = stats.pearsonr(yearly_resample.dropna().precipitation, yearly_resample.dropna().total_water_consumption)

	print(f'The pearson r test shows a result of {p} and an r value of {corr}.\nSee graph below.')

	sns.lmplot(data = yearly_resample.dropna(), y='total_water_consumption', x ='precipitation')

def usage_elevation_yearly_corr_test(train, usage):
	yearly_resample = train.copy()
	yearly_resample = yearly_resample.resample('A').mean()
	yearly_resample['total_water_consumption'] = (usage.total_consumption.resample('A').mean()).interpolate(method='polynomial', order=2).astype('int64')

	# Running a pearson-r correlation test, printing out if the result is statistically significant and the r correlation value.
	corr, p = stats.pearsonr(yearly_resample.dropna().water_level_elevation, yearly_resample.dropna().total_water_consumption)

	print(f'The pearson r test shows a result of {p} and an r value of {corr}.\nSee graph below.')

	sns.lmplot(data = yearly_resample.dropna(), x='total_water_consumption', y ='water_level_elevation')

def pop_elevation_yearly_corr_test(train, pop):
	yearly_resample = train.copy()
	yearly_resample = yearly_resample.resample('A').mean()
	yearly_resample['population'] = (pop.resample('A').mean()).interpolate(method='polynomial', order=2).astype('int64')

	# Running a pearson-r correlation test, printing out if the result is statistically significant and the r correlation value.
	corr, p = stats.pearsonr(yearly_resample.dropna().water_level_elevation, yearly_resample.dropna().population)

	print(f'The pearson r test shows a result of {p} and an r value of {corr}.\nSee graph below.')

	sns.lmplot(data = yearly_resample.dropna(), y='water_level_elevation', x ='population')