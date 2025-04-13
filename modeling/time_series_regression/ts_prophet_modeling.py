import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def create_student_dataframes(student_sample):

    """
    Captures the variables of interest which are ID, date, height, weight, BMI, COI, and DMSP and formulates the data into proper time-series format

    Args:
      student_sample: sample of the students

    Returns:
      student_dfs: a reconstructed dataframe that follows time-series patterns
    """
 
    
    student_dfs = {}

    for index in student_sample.index:
        student_id= student_sample.loc[student_sample.index==index,"id"]
        student_id= student_id.values.tolist()[0]

        #making a list of all the column names we are interested in
        dcoll_list=['dcoll_{}'.format(num) for num in range(1,12)] 
        #collects date data for chosen index in dataframe
        dates_loc=student_sample.loc[student_sample.index==index, dcoll_list] 
        # converts dataframe to list
        dates_loc= dates_loc.values.tolist()[0] 

        ht_list=["ht_in_{}".format(num) for num in range(1,12)]
        ht_loc=student_sample.loc[student_sample.index==index, ht_list]
        ht_loc=ht_loc.values.tolist()[0] 

        wt_list=["wt_lbs_{}".format(num) for num in range(1,12)]
        wt_loc=student_sample.loc[student_sample.index==index, wt_list]
        wt_loc=wt_loc.values.tolist()[0] 

        BMI_list=["BMI_{}".format(num) for num in range(1,12)]
        BMI_loc=student_sample.loc[student_sample.index==index, BMI_list]
        BMI_loc=BMI_loc.values.tolist()[0] 

        COI_list=["COI_nat_{}".format(num) for num in range(1,12)]
        COI_loc=student_sample.loc[student_sample.index==index, COI_list]
        COI_loc=COI_loc.values.tolist()[0] 

        DMSP_light=["DMSP_mean_{}".format(num) for num in range(1,12)]
        DMSP_loc=student_sample.loc[student_sample.index==index, DMSP_light]
        DMSP_loc=DMSP_loc.values.tolist()[0] 

        #Formulates proper dataframe that can be inserted into models
        student_dfs['df_{}'.format(student_id)] = pd.DataFrame(data={
            'dates':dates_loc,
            'heights':ht_loc,
            'weights':wt_loc,
            'BMI':BMI_loc,
            'COI':COI_loc,
            'DMSP':DMSP_loc
        })


       
    return student_dfs




def adjusting_df_prophet(student_dfs):

    """
    Does additional processing needed for the Prophet method. The prophet methods requires columns names that follow "ds" and "y"
    
    Args:
      student_dfs: sample of the students that have been processed for time-series using create_student_dataframes function

    Returns:
      student_dfs: a reconstructed dataframe that follows time-series patterns and is prepared for Prophet model

    """


    for key in student_dfs: 
        #rename into Prophet format
        student_dfs[key]=student_dfs[key].rename(columns={"dates":"ds", "heights":"y"}) 
        #Convert ds from object to datetime
        student_dfs[key]["ds"]= pd.to_datetime(student_dfs[key]["ds"]) 
        #ds is required to be index for resampling
        student_dfs[key]=student_dfs[key].set_index("ds") 
        student_dfs[key] = student_dfs[key].resample('D', convention = "end").asfreq()
        student_dfs[key] = student_dfs[key].interpolate(method = "linear")
    
    return student_dfs




def prophet_for_height_allregressors(student_dfs):

    """
    Creates a time-series regression model using the Prophet method and considers health data as well as COI and DMSP data. It takes in the first 4 years and predicts on the 5th year; the mean absolute percentage error is created and plots are generated    
   
    Args:
      student_dfs: student data that is processed for time-series prophet modeling

    Returns:
      plt: Plots showcasing the Prophet model versus actual test path
      prophet_mapes_additional: Mean Absolute Percentage Error scores 
    """
    

    prophet_mapes_additional = []


    for key in student_dfs:
        df_train = student_dfs[key][:1463] 
        df_test = student_dfs[key][~student_dfs[key].index.isin(df_train.index)] 
        #ds cannot be index for prophet method
        df_train.reset_index(inplace=True) 
        df_test.reset_index(inplace=True)
        
        #building model
        m = Prophet() 
        m.add_regressor("COI")
        m.add_regressor("DMSP")
        m.fit(df_train)

        forecast = m.predict(df_test.drop(columns="y"))
        prophet_mapes_additional.append(mean_absolute_percentage_error(df_test['y'], forecast["trend"]))

        fig=m.plot(forecast)
        plt.plot(df_test['ds'], df_test["y"], label='test', color="orange")
        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)

    return prophet_mapes_additional  



def prophet_for_height_noregressors(student_dfs):

    """
    Creates a time-series regression model using the Prophet method and considers only health data. It takes in the first 4 years and predicts on the 5th year; the mean absolute percentage error is created and plots are generated    
   
    Args:
      student_dfs: student data that is processed for time-series prophet modeling

    Returns:
      plt: Plots showcasing the Prophet model versus actual test path
      prophet_mapes_additional: Mean Absolute Percentage Error scores 
    """

    prophet_mapes = []

    for key in student_dfs:
        #setting up train which is first 4 years
        df_train = student_dfs[key][:1463] 
        #test is last year of study
        df_test = student_dfs[key][~student_dfs[key].index.isin(df_train.index)] 

        #ds cannot be index for prophet method
        df_train.reset_index(inplace=True) 
        df_test.reset_index(inplace=True)
        
        #building model
        m = Prophet() 
        m.fit(df_train)

        forecast = m.predict(df_test.drop(columns="y"))
        prophet_mapes.append(mean_absolute_percentage_error(df_test['y'], forecast["trend"]))

        fig=m.plot(forecast)
        plt.plot(df_test['ds'], df_test["y"], label='test', color="orange")
        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)

    return prophet_mapes  