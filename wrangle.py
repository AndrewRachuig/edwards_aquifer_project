import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from re import sub


def get_dataframes():
    '''
    This function pulls all local csv files needed for this project and puts them into dataframes. It then returns them all out.

    Returns: dataframes for the following: aquifer, temps, precip, pop, usage
    '''
    aquifer = pd.read_csv('data_files/aquifer_j17_well_data.csv', parse_dates=[1])
    temps = pd.read_csv('data_files/monthly_avg_temp_bexar_county.csv', parse_dates=[8])
    precip = pd.read_csv('data_files/monthly_total_precipitation_bexar_county.csv', parse_dates=[8])
    pop = pd.read_csv('data_files/population_by_year_bexar_county.csv', parse_dates=[0])
    usage = pd.read_csv('data_files/water_use_bexar_county_1984_2019.csv', parse_dates=[0])
    return aquifer, temps, precip, pop, usage

# Making a snake case fixer
def snake_case(s):
    '''This function takes in a word or phrase and puts it into snake case.
    
    Returns: a snake case version of the input word or phrase
    '''
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
        sub('([A-Z]+)', r' \1',
        s.replace('-', ''))).split()).lower()

# Making a df column renamer using the snake_case function above.
def column_renamer(df):
    '''
    This function takes in a dataframe and transforms all column names into snake case.

    Returns: The original dataframe with newly made snake case column names.
    '''
    new_names = {}
    for col in df.columns.to_list():
        new_names[col] = snake_case(col)
    renamed = df.rename(columns = new_names)
    return renamed

def clean_all_dataframes(aquifer, temps, precip, pop, usage):
    '''This function take in all of the previously made dataframes and performs various cleaning operations on them.
    Actions performed will be commented below.  It then returns out all the cleaned dataframes. As a note, this returns out a combined
    dataframe of precip and temps called weather
    
    Parameters: aquifer - the aquifer dataframe previously pulled in
                temps - the temperature dataframe previously pulled in
                precip - the precipitation dataframe previously pulled in
                pop - the population dataframe previously pulled in 
                usage - the water usage dataframe previously pulled in
                
    Returns: aquifer, weather, pop, usage as cleaned versions of their previously dirty dataframes'''
    #Cleaning aquifer by dropping unused column, renaming columns and setting the datetime as the index
    aquifer = aquifer.drop('Site', axis=1)
    aquifer = aquifer.rename(columns = {'DailyHighDate': 'date', 'WaterLevelElevation': 'water_level_elevation'}).set_index('date').sort_index()


    #Cleaning temps by dropping unused columns, renaming columns and setting the datetime as the index
    temps = temps.rename(columns = {'DATA 0': 'avg_monthly_temp', 'TIME 0': 'date'})
    temps = temps[['avg_monthly_temp', 'date']]
    temps = temps.set_index('date').sort_index()

    #Cleaning precip by dropping unused columns, renaming columns and setting the datetime as the index
    precip = precip.rename(columns = {'DATA 0': 'total_monthly_precip', 'TIME 0': 'date'})
    precip = precip[['total_monthly_precip', 'date']]
    precip.set_index('date').sort_index()

    #Cleaning pop by dropping unused columns, renaming columns and setting the datetime as the index
    pop = column_renamer(pop)
    pop = pop.rename(columns = {'year':'date'})
    pop = pop[['date', 'population']]
    pop = pop.set_index('date').sort_index()

    #Cleaning usage by dropping unused columns, renaming columns and setting the datetime as the index; also makes a new column 
    # called total_consumption
    usage = column_renamer(usage)
    usage = usage.rename(columns = {'year':'date'})
    usage = usage.set_index('date').sort_index()
    usage['total_consumption'] = usage.municipal + usage.manufacturing + usage.mining + usage.power + usage.irrigation + usage.livestock

    # Making a merged dataframe of temps and precip
    weather = temps.merge(precip, on ='date')
    weather = weather.set_index('date').sort_index()

    return aquifer, weather, pop, usage

def train_test_split(aquifer):
    '''
    This function takes in a prepared and cleaned aquifer dataframe and splits it in preparation for EDA and modeling. It displays a plot of the distribution
    of the split and returns train and test dataframes.

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