import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
import matplotlib

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import month_plot

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def create_student_dfs(data):
    """
    This function creates columns for collection dates, heights, BMIs,
    COI values, and DMSP values for streamlined time series modeling.

    Args:
        - data: a DataFrame containing student health data

    Returns: a dictionary containing several students' health data
    """

    # Initialize dictionary to store DataFrames
    student_dfs = {}

    # Populating DataFrame columns
    for idx in data.index:
        student_id = data.loc[data.index == idx, 'id']
        student_id = student_id.values.tolist()[0] 

        dcoll_list = ['dcoll_{}'.format(num) for num in range(1, 12)]
        dates_loc = data.loc[data.index == idx, dcoll_list]
        dates_loc = dates_loc.values.tolist()[0]

        ht_list = ['ht_in_{}'.format(num) for num in range(1, 12)]
        ht_loc = data.loc[data.index == idx, ht_list]
        ht_loc = ht_loc.values.tolist()[0]

        bmi_list = ['BMI_{}'.format(num) for num in range(1, 12)]
        bmi_loc = data.loc[data.index == idx, bmi_list]
        bmi_loc = bmi_loc.values.tolist()[0]

        coi_list = ['COI_nat_{}'.format(num) for num in range(1, 12)]
        coi_loc = data.loc[data.index == idx, coi_list]
        coi_loc = coi_loc.values.tolist()[0]

        dmsp_list = ['DMSP_mean_{}'.format(num) for num in range(1, 12)]
        dmsp_loc = data.loc[data.index == idx, dmsp_list]
        dmsp_loc = dmsp_loc.values.tolist()[0]

        key = 'df_{}'.format(student_id)
        
        student_dfs[key] = pd.DataFrame(data = {
            'd_coll': dates_loc,
            'heights_in': ht_loc,
            'BMI': bmi_loc,
            'COI_nat': coi_loc, 
            'DMSP_mean': dmsp_loc
        })

        student_dfs[key].index = pd.to_datetime(student_dfs[key]['d_coll'])
        student_dfs[key].drop(columns = 'd_coll', inplace = True)

        # Resampling to create daily frequency and 
        # interpolating to fill in missing data
        student_dfs[key] = student_dfs[key].resample('D', convention = "end").asfreq()
        student_dfs[key] = student_dfs[key].interpolate(method = "linear")

    return student_dfs

def plot_student_growth(data_dict, var_list):
    """
    This function displays plots of students' heights 
    and BMIs over time.

    Args:
        - data_dict (dict): a dictionary containing several DataFrames
        - var_list (list): a list of variables to plot over time
    """

    for var in var_list:
        for key in data_dict:
            data_dict[key][var].plot()
            plt.xlabel("Time")
            plt.ylabel(var)
            plt.title("{} over Time".format(var))
        plt.show()

def seasonal_subseries(data_dict, var_list):
    """
    This function displays seasonal subseries plots for certain
    variables associated with several students.

    Args:
        - data_dict (dict): a dictionary containing several DataFrames
        - var_list (list): a list of variables to plot over time
    """
    
    for key in data_dict:
        data = data_dict[key]
        data = data.resample('1M').mean()
        
        fig, ax = plt.subplots(nrows=1, ncols=len(var_list), figsize=(16, 6))
        i = 0
        for var in var_list:
            month_plot(data[var], ylabel = var, ax=ax[i])
            i += 1

def plot_decomposition(data_dict, var_list):
    """
    This function displays seasonal decomposition plots
    for time series data.

    Args:
        - data_dict (dict): a dictionary containing several DataFrames
        - var_list (list): a list of variables to plot over time
    """

    for var in var_list:
        for key in data_dict:
            df = data_dict[key][var]
            student_decompose = seasonal_decompose(df, model = 'additive', period = 182)
            student_decompose.plot()



def adfuller_test(data_dict, var_list):
    """
    This function displays p-values from the ADFuller test
    to determine stationarity. P-values greater than 0.05
    indicate that the series is stationary.

    Inputs:
        - data_dict (dict): a dictionary containing several DataFrames
        - var_list (list): a list of variables to plot over time
    """

    p_vals = {}
    for var in var_list:
        p_vals[var] = [] 

    for key in data_dict:
        df = data_dict[key]
        for y in var_list:
            adf_test = adfuller(df[y], autolag = 'AIC')
            p_vals[y].append(adf_test[1])

    print('p-values:')
    print('___________')
    print(p_vals)

def sarima_modeling(data_dict, dep_variable):
    """
    This function generates SARIMA models given a 
    data dictionary and a dependent variable.

    Args:
        - data_dict (dict): a dictionary containing DataFrames
        - dep_variable (str): the column/variable to model

    Returns: a dictionary containing a list of MAPE and 
             a list of RMSE values.
    """
    
    # Defining dictionary of metrics.
    sarima_metrics = {'sarima_mape': [],
                      'sarima_mse': [],
                      'sarima_rmse': []}
    
    # Generating SARIMA models for each DataFrame.
    for key in data_dict:
        df = data_dict[key]

        # Splitting into train and validation set
        df_train = df[:1463]
        df_valid = df[~df.index.isin(df_train.index)]

        samodel = pm.auto_arima(df_train[[dep_variable]],
                                start_p=1, start_q=1,
                                test='adf',
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=None, D=1, trace=False,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True, n_jobs = -1)

        fc = samodel.predict(len(df_valid))  # 95% conf


        # Plot
        params = {'axes.labelsize': 15,'axes.titlesize':15, 'legend.fontsize': 15, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
        plt.rcParams.update(params)
        plt.figure(figsize=(12,5), dpi=100)
        #plt.rcParams.update({'font.size': 50})
        plt.plot(df_train[dep_variable], label='train', linewidth = 3)
        plt.plot(df_valid[dep_variable], label='test', linewidth = 3)
        plt.plot(fc, label='forecast', linewidth = 3)
  
        plt.title('SARIMA - Forecast vs Actuals: {}'.format(dep_variable))
        plt.xlabel('Year')
        plt.ylabel(dep_variable)
        plt.legend(loc = 'upper left')
        plt.show()


        sarima_metrics['sarima_mape'].append(mean_absolute_percentage_error(df_valid[dep_variable], fc))
        sarima_metrics['sarima_mse'].append(mean_squared_error(df_valid[dep_variable], fc))
        sarima_metrics['sarima_rmse'].append(np.sqrt(mean_squared_error(df_valid[dep_variable], fc)))

    return sarima_metrics

def sarimax_modeling(data_dict, dep_variable, exog_variable):
    """
    This function generates SARIMAX models given a 
    data dictionary, dependent variable, and exogenous
    variable.

    Args:
        - data_dict (dict): a dictionary containing DataFrames
        - dep_variable (str): the column/variable to model
        - exog_variable (str): an exogenous variable

    Returns: a dictionary containing a list of MAPE and 
    a list of RMSE values.
    """

    sarimax_metrics = {'sarimax_mape': [],
                       'sarimax_mse': [],
                       'sarimax_rmse': []}
    for key in data_dict:
        # sample df
        df = data_dict[key]

        df_train = df[:1463]
        df_test = df[~df.index.isin(df_train.index)]
        # SARIMAX Model
        sxmodel = pm.auto_arima(df_train[[dep_variable]], exogenous=df_train[[exog_variable]],
                                start_p=1, start_q=1,
                                test='adf',
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=None, D=1, trace=False,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True, n_jobs = -1)

        fc = sxmodel.predict(len(df_test), 
                             exogenous = np.array(df_test[exog_variable])) 

        # Plot
        params = {'axes.labelsize': 15,'axes.titlesize':15, 'legend.fontsize': 15, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
        plt.rcParams.update(params)
        plt.figure(figsize=(12,5), dpi=100)
        plt.plot(df_train[dep_variable], label='train', linewidth = 3)
        plt.plot(df_test[dep_variable], label='test', linewidth = 3)
        plt.plot(fc, label='forecast', linewidth = 3)
        # plt.fill_between(conf_int.index, conf_int['lower heights_in'], conf_int['upper heights_in'], 
        #                  color='k', alpha=.15)
        plt.title('SARIMAX - Forecast vs Actuals: {}'.format(dep_variable))
        plt.legend(loc = 'upper left', fontsize=8)
        plt.show()

        sarimax_metrics['sarimax_mape'].append(mean_absolute_percentage_error(df_test[dep_variable], fc))
        sarimax_metrics['sarimax_mse'].append(mean_squared_error(df_test[dep_variable], fc))
        sarimax_metrics['sarimax_rmse'].append(np.sqrt(mean_squared_error(df_test[dep_variable], fc)))

    return sarimax_metrics
