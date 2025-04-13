This is an (nearly) empty data directory.

Required files:
   - "student_data.csv" Fort Bend student data (secure)
   - "trajgps.csv" — student trajectory group labels (secure)
   - "tracts.xlsx" — school-to-census-tract mappings
   - "coi_data_cleaned.csv" — COI data
   - "dmsp.csv" — ALAN data

To download the data, contact the Baylor College of Medicine team or Rice D2K student team administrators to be granted access to the relevant folder in Box: "BCM Growth Trajectories (common files) > Datasets". Please note that all student heath data are sensitive and confidential; you may need to earn your human subjects research certificates or prove your certifications are up-to-date. There are also restrictions on where the data may be stored and uploaded. They are not to be shared with any outside party without explicit consent from BCM study administrators.

Once you have downloaded the data files, save them to the "data_directory" folder.

"file_paths.txt" contains file names for loading data into R. These file names may be edited to match those on your local device, as needed, before running "clean_student_data_first.RMD". 

Once these steps have been completed, you are ready to move to data cleaning and preprocessing (directory folder: "cleaning_preprocessing").