# The file includes function(s) for cleaning student data 
# after it has been run through the "clean_student_data_first.RMD" file

# Be sure necessary packages are installed and correctly versioned!
# Essential libraries
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

#main cleaning function
def clean_student_data(local_dir, 
                       student_file = "growth.csv",
                       tracts_file = "tracts.xlsx",
                       traj_group_file = "trajgps.csv",
                       complete_only = True):
    """
    Clean student data (from "clean_student_data_first.RMD" output). 
    The new output data features the following changes:
        - extraneous columns removed;
        - added columns bmipbool_[1-11], where 0 is if a child's BMI percentile 
            is less than 85 and 1 otherwise
        - added column 'school_mode', the code for a child's most-attended school;
        - added column 'school_mode_tract', the school_mode's census tract;
        - added column 'is_complete', a Boolean series for complete case data;
        - added column 'is_train', a Boolean series for an 80-20 train-test data 
            split along school lines (using school_mode);
        - added column 'GROUP', the trajectory group associated with
            each observation (child) in the Moreno lab's work.

    Args:
        local_dir (string): an individual's local working directory path
        student_file (string, optional): file name of student data 
                                            (must be *.csv)
        tracts_file (string, optional): file name of school-tract data 
                                            (must be *.xlsx)
        traj_group_file (string, optional): file name of trajectory group data 
                                                (must be *.csv)
        complete_only (Boolean): True = output complete case data only,
                                 False = output all data
    
    Returns:
        student_clean (dataframe): cleaned student dataframe
    """
    #import student data
    student1 = pd.read_csv(str(local_dir) + "/" + str(student_file))

    #Remove extraneous columns
    student1.reset_index(inplace = True)
    student1.drop(columns = ['Unnamed: 0'], inplace = True)
    student1.drop(columns = ['index'], inplace = True)

    # create columns bmipbool_\d+, where 0 is if bmi percentile of 
        # child is less than 85, 1 otherwise, nan if missing bmi percentile data
    for i in range(1,12):
        student1['bmipbool_' + str(i)] = np.where(student1['bmip_' + str(i)]<85, 0, 1)
        student1['bmipbool_' + str(i)] = \
            np.where(np.isnan(student1['bmip_' + str(i)]), np.nan, student1['bmipbool_' + str(i)])

    #import schools and census tracts
    tracts = pd.read_excel(str(local_dir) + "/" + str(tracts_file))

    ## Clean and Merge Data
    
    # We retrieve the school mode (the most-attended school among the 11 
    # measurement points per student). If a student has two or more modes, 
    # we choose the first mode, due to the greater relative importance of 
    # early-childhood neighborhood factors (Galster, 2011; Minh et al., 2017).

    #get all school columns
    school_df = student1.filter(regex = "^school_\d+$")

    #get mode for school
    student2 = student1.assign(
                             school_mode = school_df.mode(axis = 'columns')[0]
                            ) 

    # We also want the census tract associated with each student's 
    # "school mode," for the purposes of modeling, model validation, 
    # and visualization.

    #select and rename school-tract columns
    tracts_join = tracts[['Number', 'TRACT CODE']].rename(columns=
                {"Number":"school_mode", "TRACT CODE": "school_mode_tract"})

    #join census tract (school_mode_tract) to school mode
    student3 = student2.merge(tracts_join, on = 'school_mode', how = 'left')

    ## Add Moreno trajectory groups

    #import trajectory data
    traj_init = pd.read_csv(str(local_dir) + "/" + str(traj_group_file))

    #select relevant variables
    traj_int = traj_init[['id', 'GROUP']]

    #get trajectory group mode over each student's observations (in case of changes)
    traj = traj_int.groupby('id')['GROUP'].agg(
        pd.Series.mode).to_frame()

    #left join trajectory groups onto cleaned student data
    student4 = student3.merge(traj, on = 'id', how = 'left')

    # Label rows without any null values (select only complete cases)
    student4['is_complete'] = ~student4.isna().any(axis = 1) 

    ## Split Data

    # We split the data along an approximate 80:20 ratio, where 80% of the data 
    # will be used as training data and the remaining 20% will be used as test 
    # data. This is a rule of thumb by many data science practitioners and is 
    # based on the Pareto principle (Joseph, 2022).

    # We have dependence structures in our data, both hierarchical and spatial. 
    # This indicates a simple random split is not appropriate, as students in 
    # the same geographic area and school who are in the training and test data 
    # may violate the assumption of independence between the two groups (Roberts, 
    # 2017). Instead, as with our blocked cross-validation, we perform a random 
    # test-train split but preserve the cohesion of each school.
 
    # We create a random split with appropriate proportions along school lines
    # first for complete cases only, then for the remaining data. We aim that
    # the train-test split can be used with either the complete-data or the 
    # all-data case.
 
    student_clean = get_tt_split(student4, 0.8)

    # reset the indeces
    student_clean.index = pd.RangeIndex(len(student_clean.index))

    student_clean.index = range(len(student_clean.index)) 
    print(student_clean)



    if complete_only:
         return student_clean[student_clean['is_complete']]
    else:
        return student_clean



def get_tt_split(input_data, prop_train = 0.8):
        """
        Get train-test split with desired proportions, along nesting variable
        (school) lines. Split will have correct proportions for complete-case subset
        and for the entire sample.

        Args:
            input_data (dataframe): student data with columns 'id', 'is_complete',
                                    and 'school_mode'
           prop_train (float): desired proportion of data to be set aside as training data
    
        Returns:
           output_data (dataframe): student data with desired train-test split
        """
        #cycle through random seeds that will split data 80:20 for both schools and students
        for i in range(1,1000):
            random.seed(a = i)

            #get schools that actually appear in school_mode 
            schools_used = input_data['school_mode'].unique()

            #generate boolean vector with k = number of unique schools in school_mode
            school_bools = random.choices([True, False], k = len(schools_used),
                                        weights = [prop_train, 1 - prop_train])

            #link schools to boolean values
            schools_with_bools = pd.DataFrame({"school_mode": schools_used,
                                           "is_train": school_bools})

            #join to main student dataframe
            output_data = input_data.merge(schools_with_bools, 
                                               on = 'school_mode', how = 'left')
    
            #check what proportion of student data is used for training
            #with current random seed
            output_data_complete = output_data[output_data['is_complete']]
            prop_train_complete = (output_data_complete.groupby(['is_train']).size() / len(output_data_complete))[1]
            prop_train_all = (output_data.groupby(['is_train']).size() / len(output_data))[1]

            #check whether the proportion set aside for training 
            #falls within acceptable parameters (within 0.01 of prop_train)
            correct_prop = prop_train_complete > (prop_train - 0.01) and \
                           prop_train_complete < (prop_train + 0.01) and \
                           prop_train_all > (prop_train - 0.01) and \
                           prop_train_all < (prop_train + 0.01)
            
            
            #if train-test split acceptable, end loop
            if correct_prop:
                break 
    
        return(output_data)



# dev note: compiled from 'data_split.ipynb', 'coyle_cleaned_data_indicators.ipynb',
    # and 'oudekerk_traintest_split.ipynb'