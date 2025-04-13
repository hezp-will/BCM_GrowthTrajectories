import pandas as pd


def melt_by_var(var_list, var_name, data, student_id_col, time_id_col):
  """
  Melt the dataset based on a variable and the list of the columns that measure that variabled throughout time

  Args:
    var_list(list) = list of all column name that represent different time measurements of a variable
    var_name (string) = name of the variable that is measured
    data (dataframe) = dataset that includes the students ID and the columns listed
    studnet_id_col(string) = column in current dataframe that contains ID for each student
    time_id_col(string) = new desired name for ID for each student and time measurement

  Returns:
   n_var_melt(dataframe): melted dataframe to have one observation per time of measure per child
  """


  #new column name to show time index
  intervals =  ["01","02","03","04","05","06","07","08","09","10","11"]

  var_time_col = var_name + "_t"

  #zip the variable names with the invervals list to create a dictionary
  var_time_pair = zip(var_list, intervals)
  var_dict = dict(var_time_pair)


  #subset data to only include student ID and columns of relevant variable
  var_df = data[[student_id_col]+ var_list]
  #melt dataset based on the variable
  var_melt = pd.melt(var_df, id_vars = student_id_col, value_vars = var_list, var_name = var_time_col, value_name = var_name)

  #replace each column with the variable time
  n_var_melt = var_melt.replace({var_time_col:var_dict})

  #create a new ID column
  n_var_melt[time_id_col]= n_var_melt[student_id_col].astype(str)+n_var_melt[var_time_col].astype(str)
  return n_var_melt


def merge_melt(var_dict, data, student_id_col,time_id_col):
  """
  Create sub-dataframes that melt the dataframe by list of variables and merges all sub-dataframes together. 
  Args:
    var_dict(dict): dictionary of all the columns where key are variable names, and values are a list of all columns for each variable that neeed to be melted
    data (dataframe) = dataset that includes the students ID and the columns listed
    student_id_col(string) = column in current dataframe that contains ID for each student
    time_id_col(string) = new desired name for ID for each student and time measurement

  Returns:
    merge_data(dataframe): melted and merged dataframe for all variables inputted
  """
  #create counter variable for first iteration
  empty = True
  for var in var_dict:
    #create melted dataframe for given variable
    var_melted = melt_by_var(var_dict[var], var, data, student_id_col, time_id_col)
    if empty == True:
      merge_data = var_melted
      empty = False
    else:
      #merge dataframe with prevoius melted dataframe
      merge_data = pd.merge(merge_data, var_melted, left_on = [student_id_col,time_id_col], right_on = [student_id_col,time_id_col])
  return merge_data



def find_age(dob, date, data):

  """
  Create new column in database that contains numeric age (in years) of a child on given 
  measurement point based on date of birth of child and the date of the data collection

  Args:
    dob(str): column of data of birth of each child
    data(str): column of dates for each time measure of the data collected
    data (dataframe) = dataset that includes the students ID and the columns listed

  Returns:
    data_raw_age(dataframe): data with a new column ("raw_age") that captures the age in years
  """
  #first convert column values into date formate
  data[date]= pd.to_datetime(data[date])
  data[dob]= pd.to_datetime(data[dob])
  data["raw_age"]= (data[date]-data[dob])/ pd.Timedelta (days = 365)

  return data




def melt_data(all_df):
  """
  Modify the dataframe from wide format to long format. Resulting dataset is exported the data directory.

  Args:
    all_df (database): raw intitial student database
  """
  #create demographic variables to keep for each student
  #_____________________________________
  var_dict = {}
  #list of DMSP columns
  var_dict["DMSP"] =['DMSP_mean_1', 'DMSP_mean_2','DMSP_mean_3','DMSP_mean_4','DMSP_mean_5','DMSP_mean_6','DMSP_mean_7','DMSP_mean_8','DMSP_mean_9','DMSP_mean_10','DMSP_mean_11']
  #list of COI columns
  var_dict["COI"] =['COI_nat_1','COI_nat_2','COI_nat_3','COI_nat_4','COI_nat_5','COI_nat_6','COI_nat_7','COI_nat_8','COI_nat_9','COI_nat_10','COI_nat_11']
  #list of census tract columns
  var_dict["ctract"] = ["tract_1","tract_2","tract_3","tract_4","tract_5","tract_6","tract_7","tract_8","tract_9","tract_10","tract_11"]
  #list of colums for BMI's
  var_dict["BMI"] = ["BMI_1","BMI_2","BMI_3","BMI_4","BMI_5","BMI_6","BMI_7","BMI_8","BMI_9","BMI_10","BMI_11"]
  #list of columns for age:
  var_dict["date"]  = ["dcoll_1","dcoll_2","dcoll_3", "dcoll_4","dcoll_5","dcoll_6","dcoll_7","dcoll_8","dcoll_9","dcoll_10","dcoll_11"]
  #list for BMIz columns
  var_dict["BMIZ"] = ["bmiz_1","bmiz_2","bmiz_3","bmiz_4","bmiz_5","bmiz_6","bmiz_7","bmiz_8","bmiz_9","bmiz_10","bmiz_11"]
  #list for BMI_p_bool
  var_dict["BMI_bool"]= ["bmipbool_1","bmipbool_2","bmipbool_3","bmipbool_4","bmipbool_5","bmipbool_6","bmipbool_7","bmipbool_8","bmipbool_9","bmipbool_10","bmipbool_11"]
  #_____________________________________
  #defining constants 
  dem_var = ["ethnic", "sex", "dob", "id", "COI_cat", "school_mode"]

  #create COI columns
  #creating variable that captures the average COI per student
  all_df['COI_average'] = all_df[var_dict["COI"]].mean(axis=1)
  #convert numerica values into factor categories
  all_df["COI_cat"] = pd.cut( x= all_df["COI_average"], bins=[0,20,40,60,80,100], labels = ["Very Low", "Low", "Moderate", "High", "Very High"],)
  #create dataframe with demographic variable
  dem_df = all_df[dem_var]
  #melt the dataset based on variable dictionary 
  melt_var = merge_melt(var_dict,all_df,"id", "student_time_id")
  #merge melted variable dataset with demographic dataset 
  melt_dem_df = pd.merge(melt_var, dem_df, on= "id") 
  #create age column 
  melt_final =find_age("dob", "date", melt_dem_df) 
  melt_final.to_csv("../../data_directory/melt_final.csv", index = False)
  
  
  
