import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error

plt.rcParams["figure.figsize"] = (14, 8)
import wrangle

def splits(aquifer):
    '''
    This function takes in a prepared and cleaned aquifer dataframe and splits it in preparation modeling. The train size is 60% of the total data.
    The validate size is 25% of the total data. The test size is 15% of the total data. The function also displays a plot of the distribution of the split and 
    returns train, validate and test dataframes.

    Parameters: aquifer - the aquifer dataframe previously pulled in and cleaned

    Returns:    train - dataframe of the train set of aquifer ready for modeling
                validate - datafrmae of the validate set of aquifer ready for modeling
                test - dataframe of the test set of aquifer ready for modeling
    '''

    # Setting train size to be 60% of the total dataset
    train_size = int(round(aquifer.shape[0] * 0.6))

    # set validate size to be 25% of the total dataset
    validate_size = int(round(aquifer.shape[0] * 0.25))

    # Setting test size to be 15% of the total dataset. 
    test_size = int(round(aquifer.shape[0] * 0.15))

    # Setting an variable that stores the location of the end of the validate split
    validate_end_index = train_size + validate_size

    # Making the splits
    train = aquifer[:train_size]
    validate = aquifer[train_size:validate_end_index]
    test = aquifer[validate_end_index:]

    # Checking to make sure the split worked
    if len(aquifer) == train_size + validate_size + test_size:
        print('The lengths of train, validate, test match the total length of the dataframe. Ready for modeling.')
    else:
        print('Something went wrong. The lengths of train, validate, test do not match the total length of the dataframe.')

    # Setting the datasets to have accurate frequency data
    train = train.asfreq('d', method='bfill')
    validate = validate.asfreq('d', method='bfill')
    test = test.asfreq('d', method='bfill')

    #Plotting out the splits to visualize them
    plt.plot(train, label = 'train')
    plt.plot(validate, label = 'validate')
    plt.plot(test, label = 'test')
    plt.legend()
    plt.show()

    return train, validate, test

def evaluate(validate, yhat_df, target_var):
    '''
    This function takes in the actual values of the target_var from validate, and the predicted values stored in yhat_df, 
    and computes the rmse, rounding to 0 decimal places. Finally it returns the rmse. 
    '''
    rmse = round((mean_squared_error(validate[target_var], yhat_df[target_var]))**(1/2), 0)
    return rmse

def plot_and_eval(train, validate, yhat_df, target_var, model_type):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will also lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(model_type)
    rmse = evaluate(validate, yhat_df, target_var)
    print(model_type, ' model -- RMSE: {:.0f}'.format(rmse))
    plt.show()

def append_eval_df(eval_df, validate, yhat_df, model_type, target_var):
    '''
    This function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(validate, yhat_df, target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)

def lov_model(train, validate, eval_df):
    # Grabbing the most recent observation in train and assigning it to a variable
    model_type = 'last_observed_value'
    last_observed_level = train['water_level_elevation'][-1:][0]
    yhat_df = pd.DataFrame({'water_level_elevation': [last_observed_level]}, index=validate.index)
    plot_and_eval(train, validate, yhat_df, 'water_level_elevation', model_type)
    eval_df = append_eval_df(eval_df, validate, yhat_df, model_type, target_var = 'water_level_elevation')
    return eval_df