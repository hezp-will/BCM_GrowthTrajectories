# D2K Capstone Project: BCM Growth Trajectories, Fall 2023
## Investigating the Impact of Neighborhood Equity and Exposure to Light at Night on Children's Growth Trajectories during Elementary School

The module is a codebase for a Rice D2K (Data to Knowledge) Lab capstone project, in partnership with the Baylor College of Medicine. The project investigates the impact of neighborhood-level factors, namely the Child Opportunity Index (COI) and exposure to artificial light at night (ALAN), on children's growth trajectories and probability of becoming overweight or obese. 

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [Data](#data)
3. [Installation](#installation)
   - [Dependencies](#dependencies)
      * [Python](#python-requirements)
      * [R](#r-requirements)
   - [Managing Python Environments](#managing-python-environments)
      * [Using conda](#using-conda)
4. [Demo Notebook: a Guide](#demo-notebook-a-guide)
5. [Project Tree](#project-tree)
6. [Project Team](#project-team)

## Project Objectives
* Train unsupervised algorithms, e.g., time series clustering, to identify patterns and groupings in height and BMI gain over time for children.
* Build a model to evaluate impact of neighborhood child opportunity index on seasonal patterns in children’s height and BMI gain and risk of developing overweight or obesity during elementary school.
* Examine the impact of exposure to light at night on seasonal patterns in children’s height and BMI gain and risk of developing overweight or obesity during elementary school.

## Data

The main dataset for this project, provided by Baylor College of Medicine (BCM), contains health and demographic data for approximately 7,600 Fort Bend ISD elementary-school students, collected between 2005 and 2010. It contains individual-level health data and is thus subject to privacy restrictions. Additionally, Dr. Moreno's lab has calculated BMI trajectory groups among children using these data and group-based trajectory modeling; these labels are contained in an additional file.

The student team independently collected the census tract values for each elementary school, to connect student data to neighborhood data.

The team also obtained an dataset containing COI data from "diversitydatakids.org." The dataset contains domain and census-tract-level COI values for tracts across the U.S. The team also collected ALAN measurements from image data released by the U.S. Air Force Defense Meteorological Satellite Program (DMSP), then used ArcGIS to transform this image data into quantitative data measuring the average visible light band in every census tract within Fort Bend County.

Required files to run the module:
   - "student_data.csv" Fort Bend student data (secure)
   - "trajgps.csv" — student trajectory group labels (secure)
   - "tracts.xlsx" — school-to-census-tract mappings (secure)
   - "coi_data_cleaned.csv" — COI data
   - "dmsp.csv" — ALAN data

For data download instructions, see [the data_directory README](data_directory/README.md).

## Installation

### Dependencies

The module runs on Python (version 3.11.5) and R (version 4.1.3). 

### Python requirements:

The following packages and their dependencies are required. 

* python=3.11.5
* pandas=2.0.3
* matplotlib=3.7.2
* scikit-learn=1.3.0
* tslearn=0.6.2
* prophet=1.1.4
* ipykernel=6.25.0
* openpyxl=3.0.10
* h5py=3.9.0
* pmdarima=2.0.3
* ipywidgets=8.0.4

These packages may be batch-installed by setting up a local environment using the "environment.yml" file, with instructions below. The file also contains full details of dependencies, with version, batch, and source labels.

### R requirements:

The following packages and their dependencies are required.

* R>=4.1.3
* readxl=1.4.1
* tidyverse=1.3.2
* data.table=1.14.6
* cdcanthro=0.1.1
* measurements=1.5.1
* forcats=0.5.2
* lubridate=1.9.0

To get started with R, visit [this site](https://posit.co/download/rstudio-desktop/) and follow instructions to download R and RStudio. Then simply follow the instructions in the [ data_directory README](data_directory/README.md) and [cleaning_preprocessing README](cleaning_preprocessing/README.md). The R files, once run, will automatically install the correct package versions.

### Managing Python Environments

#### Using conda

1. Install the following prerequisites:
   - [Python 3](https://www.python.org/downloads/)
   - [Anaconda](https://www.anaconda.com/download)

2. Verify that conda is successfully installed in Anaconda Prompt or a terminal:
   ```html
   conda --version
   ```
   - The conda version may also be verified:
      ```html
      conda info
      ```

3. Copy the "environment.yml" file to your source directory, or the directory indicated in your terminal.

4. Create a conda environment using the "environment.yml" file:
   ```html
   conda env create -f environment.yml
   ```
   This step may take a few minutes.

   Activate the new environment:
   ```html
   conda activate D2K_BCM_env
   ```

[Linked here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) are additional details on working with Python environments.


## Demo Notebook: a Guide

The demo.ipynb notebook contains a demo of the data science pipeline procedures implemented for this project thus far. 

Prior to running the demo notebook, this Github repo must be cloned to ensure that all necessary .py files are incorporated.

### Github Repo Cloning
1. On the main page of the repository, click **<>Code**
   
<img width="115" alt="Screenshot 2023-10-17 182941" src="https://github.com/RiceD2KLab/BCM_GrowthTrajectories_F23/assets/90566440/ac844745-8a5a-4670-9b38-1a53d81a021d">

2. Copy the repostory URL. In this example, the HTTPS URL is used.

<img width="301" alt="Screenshot 2023-10-17 183121" src="https://github.com/RiceD2KLab/BCM_GrowthTrajectories_F23/assets/90566440/502cb9f6-f8c4-4840-a822-edcb34c904af">

3. Open Git Bash. If not downloaded, use [this link](https://git-scm.com/downloads) to download Git and Git Bash.

4. Change the working directory to the desired location of the cloned directory.

5. Enter the following command to create a local clone:
```html
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
```

Once the repository has been cloned into a source-code editor (e.g., VS Code, Google Colab), select the **demo.ipynb** file, as shown below in VS Code:

<img width="386" alt="Screenshot 2023-10-18 222850" src="https://github.com/RiceD2KLab/BCM_GrowthTrajectories_F23/assets/90566440/3ba5a946-5d9d-41b6-b0c8-7b5c20cdd108">

The next steps are to be followed in the **demo.ipynb** file.

Open the Command Palette using (Ctrl + Shift + P) to select the appropriate Python interpreter by entering the command in the image below:

<img width="176" alt="Screenshot 2023-10-20 185723" src="https://github.com/RiceD2KLab/BCM_GrowthTrajectories_F23/assets/90566440/a527b1dd-cd15-48d2-a2ee-bef5bdf8398a">

Select the conda-supported interpreter:

<img width="529" alt="Screenshot 2023-10-20 185952" src="https://github.com/RiceD2KLab/BCM_GrowthTrajectories_F23/assets/90566440/9f838323-398d-4482-b87a-0515d41668a8">

The **environment.yml** file, which contains the appropriate Python libraries and versions, should be in the repository. Open a New Terminal:

<img width="438" alt="Screenshot 2023-10-20 190142" src="https://github.com/RiceD2KLab/BCM_GrowthTrajectories_F23/assets/90566440/10da4153-0d8c-4a2b-aaa3-408611821525">

Enter the following command in the terminal:
```html
conda env create -f environment.yml
```

Once the environment has been created, run the following command to activate it:
```html
conda activate D2K_BCM_env
```

Creating and activating the appropriate conda environment installs the necessary libraries and their appropriate versions for the project. Once the libraries are installed, the notebook is expected to run without errors. Note that the called functions are imported from .py files located in the various folders in the repository. The following methods are featured in the notebook:
   * Data Cleaning and Preprocessing
   * Modeling
      - K-Means Time Series Clustering
      - Time Series Regression
         - SARIMA/SARIMAX Models
         - Prophet model (with and without regressors)
      - Mixed Effects Modeling

## Project Tree
```bash
├── cleaning_preprocessing
│   ├── clean_student_data_first.RMD
│   ├── clean_student_data_second.py
│   └── README.md
├── data_directory
│   ├── README.md
│   └── file_paths.txt
├── images
│   ├── height_gbtm.png
│   ├── probow.png
│   └── README.md
├── modeling
│   ├── kmeans_ts_clustering
│   │   ├── kmeans_ts_clustering_examples.ipynb
│   │   ├── kmeans_ts_clustering.py
│   │   └── README.md
│   ├── ts_regression
│   │   ├── README.md
│   │   ├── ts_prophet_modeling.py
│   │   └── ts_sarimax_modeling.py
│   ├── mixed_effects
│   │   ├── README.md
│   │   ├── mixed_effects_cleaning_data_demo.ipynb
│   │   ├── mixed_effects_cleaning_data.py
│   │   ├── mixed_effects_modeling_demo.Rmd
│   │   └── mixed_effects_modeling.R
│   └── README.md
├── .gitignore
├── demo.ipynb
├── environment.yml
├── CITATION.cff
└── README.md
```

## Project Team

**Sponsor Mentor**: Jennette P. Moreno, Ph.D.

**DSCI 435 Faculty Mentor**: Arko Barman, Ph.D.

**D2K Mentor**: Kevin McCoy

**Team Members**:

* Zachre Andrews
* Jacob Coyle
* Caleb Huang
* Gail Oudekerk
* William Pan
* Elijah Sales
