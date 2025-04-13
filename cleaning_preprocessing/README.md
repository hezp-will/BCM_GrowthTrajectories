These files perform data cleaning, selecting relevant data features from, cleaning, and combining the following datasets.

Required files to run the module:
   - "student_data.csv" Fort Bend student data (secure)
   - "trajgps.csv" — student trajectory group labels (secure)
   - "tracts.xlsx" — school-to-census-tract mappings
   - "coi_data_cleaned.csv" — COI data
   - "dmsp.csv" — ALAN data

   For data download questions, see the data_directory README file.

Step 1:
  * Run ["clean_student_data_first.RMD"](./clean_student_data_first.RMD). It is necessarily in R, due to a CDC benchmarking algorithm (package: "cdcanthro") solely available in SAS and R.  The file outputs a data file, "growth.csv".
     * Note: if you saved your files to a folder other than the repository's data_directory folder, edit the "working_directory" on line 51 of "clean_student_data_first.RMD" to match your local file path.

Step 2:
  * Run ["clean_student_data_second.py"](./clean_student_data_second.py). It is a Python module, whose functionality is demonstrated in "demo.ipynb". Please note that to run the module, you must have "growth.csv" saved to your local directory.
