These files go through the process of applying linear mixed-effects modeling to the data. The files in this folder include:
- mixed_effects_cleaning_data.py: defines functions that clean data (primarily melting) for mixed-effects modeling
- mixed_effects_cleaning_data.ipynb: applies functions (in a demo) that clean data for mixed-effects modeling. NOTE: this file can only be run AFTER demo.ipynb is run
- mixed_effects_modeling.R: defines functions that perform mixed-effects modeling, plus model checks and interpretation
- mixed_effects_modeling_demo.Rmd: applies functions (in a demo) that perform mixed-effects modeling, etc.

Note that the data cleaning functions are in Python and the modeling functions are in R, the latter out of technical necessity (reference materials and software available for R and not available for Python).

It is necessary to apply the data cleaning functions before proceeding to the modeling step.
