import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error

from prophet import Prophet

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
    plt.plot(yhat_df[target_var], label = 'Model Predictions')
    plt.legend()
    plt.title(model_type)
    rmse = evaluate(validate, yhat_df, target_var)
    print(f'Predicting for the {model_type} model -- RMSE: {rmse}')
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

def simple_avg_model(train, validate, eval_df):
    # Grabbing the historical average of train and assigning it to a variable
    model_type = 'simple_avg'
    avg_elevation = round(train['water_level_elevation'].mean(), 2)
    yhat_df = pd.DataFrame({'water_level_elevation': [avg_elevation]}, index=validate.index)
    plot_and_eval(train, validate, yhat_df, 'water_level_elevation', model_type)
    eval_df = append_eval_df(eval_df, validate, yhat_df, model_type, target_var = 'water_level_elevation')
    return eval_df

def moving_average_model(train, validate, eval_df):
    # Grabbing the historical average of train and assigning it to a variable
    model_type = 'moving_average'

    periods = [7, 14, 30, 60, 365, 730, 1825, 3650]

    for p in periods: 
        rolling_water_levels = round(train['water_level_elevation'].rolling(p).mean()[-1], 2)
        yhat_df = pd.DataFrame({'water_level_elevation': [rolling_water_levels]}, index=validate.index)
        rmse = evaluate(validate, yhat_df, target_var = 'water_level_elevation')
        print(f"\nPredicting the {p} day Moving Average of {rolling_water_levels} -- RMSE: {rmse}")
        model_type = str(p) + '_day_moving_avg'
        for col in train.columns:
            eval_df = append_eval_df(eval_df, validate, yhat_df, model_type, target_var = 'water_level_elevation')
    
    print(f"\n\nPlot of the best Moving Average model")
    rolling_water_levels = round(train['water_level_elevation'].rolling(730).mean()[-1], 2)
    yhat_df = pd.DataFrame({'water_level_elevation': [rolling_water_levels]},
                          index=validate.index)
    plot_and_eval(train, validate, yhat_df, target_var = 'water_level_elevation', model_type = '730_day_moving_avg')

    return eval_df

def holts_model(train, validate, eval_df):
    # Creating the initial Holt's Object
    model = Holt(train, exponential=False, damped=True)
    # Fitting the Holt's object
    model = model.fit(optimized=True)
    #Making predictions for each date in validate
    yhat_items = model.predict(start = validate.index[0], end = validate.index[-1])
    # add predictions to yhat_df
    yhat_df = pd.DataFrame({'water_level_elevation': yhat_items}, index=validate.index)
    # yhat_df['water_level_elevation'] = pd.DataFrame(yhat_items)
    plot_and_eval(train, validate, yhat_df, target_var = 'water_level_elevation', model_type = 'holts_optimized')
    eval_df = append_eval_df(eval_df, validate, yhat_df, model_type = 'holts_optimized', target_var = 'water_level_elevation')
    return eval_df

def prophet_setup(train):
    # Making a new Dataframe appropriate for Prophet to use
    train_for_prophet = pd.DataFrame()
    train_for_prophet['ds'] = train.index
    train_for_prophet['y'] = train.water_level_elevation.values
    # Checking the new dataframe to confirm it looks appropriate for Prophet
    train_for_prophet.head()
    return train_for_prophet

def prophet_model(train, validate, train_for_prophet, eval_df):
    # Creating the Prophet model
    model = Prophet()
    # Fitting the model to the train_for_prophet dataframe
    model.fit(train_for_prophet)
    # Making the future dataframe to use in the next step for forecasting predictions, the periods parameter here equals
    # the length of the validate dataframe.
    length_of_validate = validate.size
    future = model.make_future_dataframe(periods = length_of_validate)
    # Forecasting/Predicting into the future of validate
    forecast = model.predict(future)
    # Plotting out the trend, as well as weekly and yearly seasonality
    # model.plot_components(forecast)
    # None

    # putting the predicted values into the yhat_df so I can use the plot and eval function to see RMSE and performance
    yhat_df = pd.DataFrame({'water_level_elevation': forecast.yhat[-7989:].values}, index=validate.index)
    # yhat_df['water_level_elevation'] = forecast.yhat[-7989:].values
    plot_and_eval(train, validate, yhat_df, target_var = 'water_level_elevation', model_type = 'facebook_prophet')
    # # Adding the Facebook Prophet model to the eval_df 
    eval_df = append_eval_df(eval_df, validate, yhat_df, model_type = 'facebook_prophet', target_var = 'water_level_elevation')
    return eval_df

def modified_prophet_model(train, validate, train_for_prophet, eval_df):
    # Creating the Prophet model
    model = Prophet(growth="flat")
    # Fitting the model to the train_for_prophet dataframe
    model.fit(train_for_prophet)
    # Making the future dataframe to use in the next step for forecasting predictions, the periods parameter here equals
    # the length of the validate dataframe.
    length_of_validate = validate.size
    future = model.make_future_dataframe(periods = length_of_validate)
    # Forecasting/Predicting into the future of validate
    forecast = model.predict(future)
    # Plotting out the trend, as well as weekly and yearly seasonality
    # model.plot_components(forecast)
    # None
    
    # putting the predicted values into the yhat_df so I can use the plot and eval function to see RMSE and performance
    yhat_df = pd.DataFrame({'water_level_elevation': forecast.yhat[-7989:].values}, index=validate.index)
    # yhat_df['water_level_elevation'] = forecast.yhat[-7989:].values
    plot_and_eval(train, validate, yhat_df, target_var = 'water_level_elevation', model_type = 'modified_facebook_prophet')
    # # Adding the Facebook Prophet model to the eval_df 
    eval_df = append_eval_df(eval_df, validate, yhat_df, model_type = 'modified_facebook_prophet', target_var = 'water_level_elevation')
    return eval_df

def mod_prophet_testing(train, validate, test, train_for_prophet):
    # Recreating the Prophet model
    model = Prophet(growth='flat')
    # Fitting the model to the train_for_prophet dataframe
    model.fit(train_for_prophet)

    # Making the future dataframe to use in the next step for forecasting predictions, the periods parameter here equals
    # equals the length of the validate dataframe.
    future = model.make_future_dataframe(periods = 12777)

    # Forecasting/Predicting into the future of validate
    forecast = model.predict(future)

    # putting the predicted values into the yhat_df so I can use the plot and eval function to see RMSE and performance
    yhat_final = test.copy()
    yhat_final['water_level_elevation'] = forecast.yhat[-4788:].values
    yhat_final['two_year_moving_avg_baseline'] = round(train['water_level_elevation'].rolling(730).mean()[-1], 2)

    # Final results plot
    plt.figure(figsize = (18, 10))
    plt.plot(train.water_level_elevation, label = 'train')
    plt.plot(validate.water_level_elevation, label = 'validate')
    plt.plot(test.water_level_elevation, label = 'test')
    plt.plot(yhat_final.water_level_elevation, label = 'FB Prophet prediction', color = 'black')
    plt.plot(yhat_final.two_year_moving_avg_baseline, label = "2 Year rolling average Baseline")
    plt.legend()
    plt.show()

    # RMSE of FB Prophet model and baseline to see results

    rmse_prophet = round(mean_squared_error(test.water_level_elevation, yhat_final.water_level_elevation)**(1/2),2)

    rmse_baseline = round(mean_squared_error(test.water_level_elevation, yhat_final.two_year_moving_avg_baseline)**(1/2),2)

    print(f'The RMSE for the FB prophet model on out-of-sample test data is:{rmse_prophet}.')
    print(f'The RMSE for the two year moving avg baseline on out-of-sample test data is:{rmse_baseline}.')