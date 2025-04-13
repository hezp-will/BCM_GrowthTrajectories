
## This is the main file containing functions for constructing, running, checking,
## validating, and plotting the linear mixed-effects models predicting children's
## BMI based on child opportunity index (COI), artificial light at night (ALAN),
## and other child- and school-level characteristics.

## For demonstration of the functions included here, see 'mixed_effects_modeling_demo.Rmd'

## Necessary R version: >= R-4.1.3

## _____________________________________________________________________________

## ENVIRONMENT AND DATA SETUP (not necessary here, but included for reference in case
##    the user is not referencing the demo)

## run the following lines to install necessary packages
# devtools::install_version("jtools", "2.2.0")
# devtools::install_version("optimx", "2023-10.21")
# devtools::install_version("lme4", "1.1-32")
# devtools::install_version("merTools", "0.6.1")
# devtools::install_version("coxed", "0.3.3")
# devtools::install_version("kableExtra", "1.3.4")
# devtools::install_version("performance", "0.10.8")
# devtools::install_version("ggh4x", "0.2.6")
# devtools::install_version("tidyverse", "1.3.2")
# remotes::install_version("Rttf2pt1", version = "1.3.8")

# run the following lines to load packages
# library(jtools)
# library(optimx)
# library(lme4)
# library(merTools)
# library(coxed)
# library(kableExtra)
# library(performance)
# library(ggh4x)
# library(tidyverse)


#set random seed for replicability
# set.seed(101)

## _____________________________________________________________________________

# DATA CLEANING

clean_training_data <- function(input_data, full_data){
  # This function cleans the training data necessary for model fitting
  #
  # Args:
  #   - input_data: a dataframe representing the students' repeated measures data
  #                 and their respective COI and ALAN values      
  #   - full_data: the Fort Bend ISD student dataset
  # Returns:
  #   - input_data_train: the cleaned training data to be used in model fitting
  
  ## Adding season variable
  # 0 for fall semester (summer season), 1 for spring semester (winter season)
  input_data <- input_data %>% mutate(season = ifelse(BMI_t %% 2 == 1, 0, 1))
  
  #convert variables into factors
  input_data <- input_data %>% mutate(season = factor(season))
   
  
  ## Making categorical variables into factors
     
  #store previous data
  input_data_old <- input_data
  
  #convert categorical variables into text factors for modeling
  input_data <- input_data %>% mutate(sex = case_when(sex == 1 ~ "male",
                                                          TRUE ~ "female"),
                                      sex = factor(sex, levels = c("male", "female")),
                                      ethnic = case_when(ethnic == 1 ~ "white",
                                                         ethnic == 2 ~ "Black",
                                                         ethnic == 3 ~ "Hispanic",
                                                         TRUE ~ "Asian"),
                                      ethnic = factor(ethnic, levels = c("Black",
                                                                         "white",
                                                                         "Hispanic",
                                                                         "Asian")),
                                      season = case_when(season == 0 ~ "fall",
                                                         TRUE ~ "spring"),
                                      season = factor(season, levels = c("fall",
                                                                         "spring")),
                                      id = as.character(id_x),
                                      school = factor(school),
                                      #rescale ALAN variable, in case needed
                                      scaled_DMSP = c(scale(input_data$DMSP)),
                                      #combine COI categories for "Low" and "Very Low"
                                      COI_cat_combined = case_when(COI_cat == "Very Low" ~ "Very Low / Low",
                                                                   COI_cat == "Low" ~ "Very Low / Low",
                                                                   TRUE ~ COI_cat),
                                      COI_cat_combined = factor(COI_cat_combined,
                                                                levels = c("Moderate",
                                                                           "Very Low / Low",
                                                                           "High",
                                                                          "Very High")),
                                      #recenter age to an intercept of 
                                      #5 years old
                                      scaled_age = raw_age - 5)
  
  #match in "train" label
  input_data_train <- 
    left_join(input_data, full_data %>% dplyr::select(id, is_train) %>%
                mutate(id = as.character(id)),
              by = "id")
  
  return(input_data_train)
}


plot_samples <- function(input_data){
  # This function generates and returns a plot containing ten students'
  # BMI trajectories over time.
  #
  # Args:
  #   - input_data: a dataframe representing the students' repeated measures data
  #                 and their respective COI and ALAN values    
  # Returns:
  #   - sample plot: a scatterplot of ten students' BMI trajectories with fitted
  #                  regression lines
  
  # Sample ten students
  sample_data = input_data[1:110, ]
  
  sample_plot <- ggplot(sample_data, aes(x = BMI_t, y = BMI, color = id))+
    geom_point() +
    geom_smooth(method = "lm", fill = NA)
  
  return(sample_plot)
}

## _____________________________________________________________________________

## MODELING

fit_coi_model <- function(input_data){
  # This function fits the input training data to a linear mixed effects model including COI.
  #
  # Args:
  #   - input_data: a dataframe representing the students' repeated measures data
  #                 and their respective COI and ALAN values
  # Returns:
  #   - bmi_model_coi: the fitted linear mixed effects model containing COI 
  #                     as a fixed effect
     
  # Fixed effects: 
  # sex (categorical)
  # ethnicity (categorical)
  # COI (categorical)
  # DMSP (continuous, radians)
  # season (categorical)
  # age (continuous)
  # interaction: COI (cat) x season (cat)
  # interaction: DMSP (cont) x season (cat)
  
  # Random effects:
  # student (id)
  # students nested within a school (note: nested vs. crossed)
  # slopes by age
  
  # (0 + raw_age|id) for random slopes by age
  # (1 | school_mode/id) represents students nested within a school
  
  
  # Model converges with bobyqa optimizer
  bmi_model_coi = lmer(BMI ~ c(sex) + c(ethnic) + c(COI_cat_combined)*c(season)*scaled_age +
                         I(scaled_age^2) +
                         (1 + scaled_age | school) + (1 + scaled_age | id),
                       data = input_data, REML = FALSE,
                       control = lmerControl(optimizer ='bobyqa'))
}

fit_dmsp_model <- function(input_data){
  # This function fits the input training data to a linear mixed effects model including DMSP.
  #
  # Args:
  #   - input_data: a dataframe representing the students' repeated measures data
  #                 and their respective COI and ALAN values
  # Returns:
  #   - bmi_model_dmsp: the fitted linear mixed effects model containing DMSP 
  #                     as a fixed effect
  
  # Fixed effects: 
  # sex (categorical)
  # ethnicity (categorical)
  # DMSP (continuous, radians)
  # season (categorical)
  # age (continuous)
  # interaction: COI (cat) x season (cat)
  # interaction: DMSP (cont) x season (cat)
  
  # Random effects:
  # student (id)
  # students nested within a school (note: nested vs. crossed)
  # slopes by age
  
  # (0 + raw_age|id) for random slopes by age
  # (1 | school_mode/id) represents students nested within a school
  
  # Model converges with bobyqa optimizer
  bmi_model_dmsp = lmer(BMI ~ c(sex) + c(ethnic) + DMSP*c(season)*scaled_age + 
                          (1 + scaled_age | school) + (1 + scaled_age | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  
  return(bmi_model_dmsp)
}

# Model summary function
model_summary <- function(lmer_model){
  # This function generates a summary of a fitted linear mixed effects model.
  #
  # Args:
  #   - lmer_model: a fitted linear mixed effects model
  # Returns:
  #   - summary: a summary of the fitted model
  #        
  summary <- summ(lmer_model)
  return(summary)
}

## _____________________________________________________________________________

## ASSUMPTION CHECKING

check_assumptions <- function(lmer_model){
  # This function generates a summary of a fitted linear mixed effects model.
  #
  # Args:
  #   - lmer_model: a fitted linear mixed effects model
  # Returns:
  #   - assumption_plot: a plot containing subplots determining model validity
  
  assumption_plot <- performance::check_model(lmer_model) 
  return(assumption_plot)
}

## _____________________________________________________________________________

## BOOTSTRAPPING FOR COEFFICIENT ESTIMATES AND CONFIDENCE INTERVALS

model_bootstrapping <- function(lmer_model, nbhd_factor, num_simulations){
  # This function generates bootstrap estimates for a given linear mixed effects
  # model.
  #
  # Args:
  #   - lmer_model: a fitted linear mixed effects model
  #   - nbhd_factor: the neighborhood-level factor to be investigated (currently
  #                   only supports "COI" or "ALAN")
  # Returns:
  #   - lmer_boot: the bootstrap coefficient estimates for the given linear mixed
  #                effects model
  

  
  ## COI model: bootstrapping for parameter estimates
  
  #function defining statistics of interest (just three parameters)
  mySumm <- function(.) { s <- sigma(.)
  c(beta =getME(., "beta"), sigma = s, sig01 = unname(s * getME(., "theta"))) }
  
  #run bootstrap (set random seed and number of simulations)
  set.seed(101)
  lmer_boot <- bootMer(lmer_model, mySumm, nsim = num_simulations,
                      seed = 1234, verbose = TRUE)
  
  # Should this line be elsewhere?
  write.csv(lmer_boot$t, paste("lmer_boot_", nbhd_factor, ".csv", sep = ""))
  
  return(lmer_boot)
}
   
create_coeff_df <- function(lmer_boot, lmer_model, nbhd_factor){
  # This function creates a dataframe containing the coefficient estimates from
  # a boostrapping procedure.
  #
  # Args:
  #   - lmer_boot: estimates generated from a bootstrapping procedure
  #   - lmer_model: a fitted linear mixed effects model
  #   - nbhd_factor: the neighborhood-level factor to be investigated (currently
  #                   only supports "COI" or "ALAN")
  # Returns:
  #   - boot_coeff_df: a dataframe containing the bootstrap estimate and confidence
  #                    intervals for fixed effects
  
  #get bootstrap confidence intervals for estimates
  #confint(boot_coi)
  lmer_boot <- lmer_boot %>% dplyr::select(-c("X", "beta1", "sigma", "sig011", "sig012",
                                            "sig013", "sig014", "sig015", "sig016"))

  if (nbhd_factor == "COI"){
    labs <- c("sex = female", "race/ethnicity = white", "race/ethnicity = Hispanic", 
                  "race/ethnicity = Asian", "COI category = Very Low, Low", "COI category = High", 
                  "COI category = Very High", "season = spring", "age (years)", "age^2",
                  "COI (Very Low, Low) * spring", "COI (High) * spring",
                  "COI (Very High) * spring", "COI (Very Low, Low) * age", 
                  "COI (High) * age", "COI (Very High) * age", "spring * age",
                  "COI (Very Low, Low) * spring * age",
                  "COI (High) * spring * age",
                  "COI (Very High) * spring * age")
  }
  else if (nbhd_factor == "ALAN") {
    labs <- c("sex = female", "race/ethnicity = white", "race/ethnicity = Hispanic", 
                   "race/ethnicity = Asian", "ALAN", "season = spring", "age (years)", 
                   "ALAN * spring", "ALAN * age", "spring * age", "ALAN * spring * age")
  }
  
  colnames(lmer_boot) <- labs
  
  boot_bca_est <- apply(lmer_boot, 2, bca)
  
  boot_t0 <- coef(summary(lmer_model))[-1,1]
  
  index_max <- nrow(data.frame(coef(summary(lmer_model)))) - 1
   
  #make dataframe of bootstrapped values for coefficients
  boot_coeff_df <- 
    data.frame(estimate = boot_t0, conf.low = boot_bca_est[1,],
               conf.high = boot_bca_est[2,], var_name = colnames(boot_bca_est)) %>%
    mutate(signif = case_when(conf.low * conf.high > 0 ~ TRUE,
                              TRUE ~ FALSE),
           index = 1:index_max,
           model = paste("Model: ", nbhd_factor, sep = ""))
  
  return(boot_coeff_df)  
}
#source for some syntax: 
#https://stackoverflow.com/questions/64268001/dot-and-whisker-coefficient-plots-using-only-mean-and-95-confidence-interval-es

create_dw_plot <- function(boot_coeff_df, nbhd_factor){
  # This function creates a dot-and-whisker plot for the coefficient estimates
  # generated from bootstrapping.
  #
  # Args:
  #   - boot_coeff_df: a dataframe containing bootstrap coefficient estimates and
  #                    confidence intervals
  #   - nbhd_factor: the neighborhood-level factor to be investigated (currently
  #                   only supports "COI" or "ALAN")
  # Returns:
  #   - dw_coeff_linear: a dot-and-whisker plot displaying the coefficient estimates
  #                      and confidence intervals
  
  boot_coeff_df_toplot <- boot_coeff_df
  
  if (nbhd_factor == "COI"){
    #manually reorder plot labels for COI model
    labs <- c("sex = female", "race/ethnicity = white", "race/ethnicity = Hispanic", 
              "race/ethnicity = Asian", "COI category = Very Low, Low", "COI category = High", 
              "COI category = Very High", "season = spring", "age (years)", "age^2",
              "COI (Very Low, Low) * spring", "COI (High) * spring",
              "COI (Very High) * spring", "COI (Very Low, Low) * age", 
              "COI (High) * age", "COI (Very High) * age", "spring * age",
              "COI (Very Low, Low) * spring * age",
              "COI (High) * spring * age",
              "COI (Very High) * spring * age")
  }
  if (nbhd_factor == "ALAN"){
    #manually reorder plot labels for ALAN model
    labs <- c("sex = female", "race/ethnicity = white", "race/ethnicity = Hispanic", 
              "race/ethnicity = Asian", "ALAN", "season = spring", "age (years)",
              "ALAN * spring", "ALAN * age", "spring * age", "ALAN * spring * age")
  }
  
  #create dot-whisker plot
  dw_coeff_linear <- 
      boot_coeff_df_toplot %>% filter(var_name != "beta1") %>%
      mutate(axis_lab = labs) %>%
      ggplot(aes(x=reorder(axis_lab,-index), y=estimate, col = signif)) +
      geom_hline(yintercept = 0, color = "gray") + 
    #plot point estimates
      geom_point(shape = 18, size = 1.75) +
    #plot confidence intervals
      geom_errorbar(aes(ymin=conf.low,ymax=conf.high),width=0)+
      coord_flip() +
      theme_minimal() +
      theme(legend.position = "none") +
      labs(x = "predictor", y = "coefficient estimate and CI",
           caption = "Baseline: Black male, intercept age = 5, measurement season = fall.",
           title = paste("Linear Mixed Model predicting BMI (version: ", nbhd_factor, ")", sep = "") +
      scale_color_manual(values = c("#440154", "#5ec962")))
      
  return(dw_coeff_linear)
}

#save visualization
#ggsave(file = "boot_dw_coeff_coi_bca.svg", plot = dw_coeff_linear,width = 7.29, height = 4.5)

# Creating combined dot-and-whisker plots

get_combined_dw <- function(boot_coi, boot_dmsp){
  # This function creates a combined dot-and-whisker plot for the coefficient estimates
  # generated from bootstrapping of both the COI and DMSP models.
  #
  # Args:
  #   - boot_coi: bootstrap coefficient estimates from the COI linear mixed effects model
  #   - boot_dmsp: bootstrap coefficient estimates from the DMSP linear mixed effects model
  # Returns:
  #   - dw_coeff_linear: a combined dot-and-whisker plot displaying the COI and
  #                      DMSP coefficient estimates and confidence intervals
  
  boot_coi <- boot_coi %>% dplyr::select(-c("X", "beta1", "sigma", "sig011", "sig012",
                                            "sig013", "sig014", "sig015", "sig016"))
  boot_dmsp <- boot_dmsp %>% dplyr::select(-c("X", "beta1", "sigma", "sig011", "sig012",
                                              "sig013", "sig014", "sig015", "sig016"))
  
  #manually reorder axis labels for COI model
  coi_labs <- c("sex = female", "race/ethnicity = white", "race/ethnicity = Hispanic", 
                "race/ethnicity = Asian", "COI category = Very Low, Low", "COI category = High", 
                "COI category = Very High", "season = spring", "age (years)", "age^2",
                "COI (Very Low, Low) * spring", "COI (High) * spring",
                "COI (Very High) * spring", "COI (Very Low, Low) * age", 
                "COI (High) * age", "COI (Very High) * age", "spring * age",
                "COI (Very Low, Low) * spring * age",
                "COI (High) * spring * age",
                "COI (Very High) * spring * age")
  
  #manually reorder axis labels for ALAN model
  dmsp_labs <- c("sex = female", "race/ethnicity = white", "race/ethnicity = Hispanic", 
                 "race/ethnicity = Asian", "ALAN", "season = spring", "age (years)", 
                 "ALAN * spring", "ALAN * age", "spring * age", "ALAN * spring * age")
  
  colnames(boot_coi) <- coi_labs
  colnames(boot_dmsp) <- dmsp_labs
  
  #get bootstrap confidence intervals for estimates
  #confint(boot_dmsp)
  boot_coi_bca_est <- apply(boot_coi, 2, bca)
  boot_dmsp_bca_est <- apply(boot_dmsp, 2, bca)
  
  #get central points from original models
  boot_coi_t0 <- coef(summary(bmi_model_coi))[-1,1]
  boot_dmsp_t0 <- coef(summary(bmi_model_dmsp))[-1,1]
  
  #make dataframe of bootstrapped values for coefficients, for ALAN model
  boot_coeff_df_dmsp <- 
    #get estimate and lower and upper confidence interval bounds
    data.frame(estimate = boot_dmsp_t0, conf.low = boot_dmsp_bca_est[1,],
               conf.high = boot_dmsp_bca_est[2,], 
               #match in variable names
               var_name = colnames(boot_dmsp_bca_est)) %>%
    #check for significance
    mutate(signif = case_when(conf.low * conf.high > 0 ~ TRUE,
                              TRUE ~ FALSE),
           #assign indices
           index = 1:11,
           model = "Model: ALAN")
  
  #make dataframe of bootstrapped values for coefficients, for COI model 
  boot_coeff_df_coi <- 
    #get estimate and lower and upper confidence interval bounds
    data.frame(estimate = boot_coi_t0, conf.low = boot_coi_bca_est[1,],
               conf.high = boot_coi_bca_est[2,], 
               # match in variable names
               var_name = colnames(boot_coi_bca_est)) %>%
    #check for significance
    mutate(signif = case_when(conf.low * conf.high > 0 ~ TRUE,
                              TRUE ~ FALSE),
           #assign indices
           index = 1:20,
           model = "Model: COI")
  
  #combine dataframes of bootstrapped values
  boot_coeff_combo <- rbind(boot_coeff_df_coi, boot_coeff_df_dmsp) %>%
    mutate(var_name = factor(var_name),
           signif = factor(signif, levels = c(TRUE, FALSE)),
           model = factor(model, levels = c("Model: COI", "Model: ALAN")))
  
  #make dot-whisker plot for coefficients
  
  #manually reorder all variables
  myorder_full <- data.frame(var_name = c("sex = female", "race/ethnicity = white", 
                                          "race/ethnicity = Hispanic", "race/ethnicity = Asian", 
                                          "age (years)", "age^2", "season = spring",
                                          "spring * age",
                                          "COI category = Very Low, Low",
                                          "COI category = High",
                                          "COI category = Very High",
                                          "COI (Very Low, Low) * age", 
                                          "COI (High) * age","COI (Very High) * age",
                                          "COI (Very Low, Low) * spring",
                                          "COI (High) * spring", 
                                          "COI (Very High) * spring",
                                          "COI (Very Low, Low) * spring * age",
                                          "COI (High) * spring * age",
                                          "COI (Very High) * spring * age",
                                          "ALAN", "ALAN * age", "ALAN * spring",
                                          "ALAN * spring * age"), order_index = 1:24)
  
  dw_coeff_linear <- 
    #join in new variable order
    left_join(boot_coeff_combo %>%
                filter(conf.low * conf.high > 0), myorder_full, by = "var_name") %>%
    ggplot(aes(x=reorder(var_name,-order_index), y=estimate, col = model)) +
    #set intercept line at zero
    geom_hline(yintercept = 0, color = "lightgray") + 
    #plot point estimates
    geom_point(shape = 18, size = 2.5) +
    #plot confidence intervals
    geom_errorbar(aes(ymin=conf.low,ymax=conf.high),width=0, linewidth = 0.75)+
    coord_flip() +
    theme_bw() +
    theme(plot.title = element_text(size = 12.5, hjust = 0.5),
          plot.subtitle = element_text(size = 9, hjust = 0.5),
          strip.text = element_text(color = "white"),
          legend.position = "none") +
    labs(x = "Predictor", y = "Coefficient Estimate and (bootstrapped) 95% CI",
         subtitle = "Baseline: Black male, intercept age = 5, measurement season = fall.",
         title = "Linear Mixed-Effects Model (predicting BMI):\nCoefficient Estimates for Significant Predictors") +
    scale_color_manual(values = c("Model: COI" = "#006D2C", "Model: ALAN" = "#08306B")) +
    #facet on model (separate COI and ALAN coefficient estimates and CIs)
    ggh4x::facet_grid2(cols = vars(model),
                       strip = ggh4x::strip_themed(
                         background_x = list(element_rect(fill = "#006D2C"),
                                             element_rect(fill = "#08306B"))))
  
  
  return(dw_coeff_linear)
  
}

## _____________________________________________________________________________

## CALCULATE AND PLOT MODEL PREDICTIONS

# get point estimate predictions and CIs

get_predictions_df <- function(input_data, nbhd_factor){
  # This function creates a dataframe containing the predictors in a linear
  # mixed effects model
  #
  # Args:
  #   - input_data: a dataframe representing the students' repeated measures data
  #                 and their respective COI and ALAN values
  #   - nbhd_factor: the neighborhood-level factor to be investigated (currently
  #                   only supports "COI" or "ALAN")
  # Returns:
  #   - predictors_df: a dataframe containing the predictors for a linear mixed
  #                    effects model
  
  #create mode function
  Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  
  #set up dataframe of example data for prediction
     
  #https://optimumsportsperformance.com/blog/making-predictions-from-a-mixed-model-using-r/
  
  # As in Aris (2022): "From these models, we predicted the population
  # average BMI over time for each neighborhood index category and plotted the corresponding
  # BMI trajectory, holding all covariates constant at their mean values.
  avg_measurement_vals <- student_data %>% group_by(BMI_t) %>% 
    dplyr::summarize(avg_age = mean(raw_age, na.rm = T),
                     avg_DMSP = mean(COI, na.rm = T)) %>% 
    mutate(season = ifelse(BMI_t %% 2 == 1, "fall", "spring"))
  
  #get average (or mode) of predictors for hypothetical new student
  if (nbhd_factor == "COI"){
    predictors_df <- 
      data.frame(#get most common values for sex and ethnicity
                 sex = Mode(student_data$sex),
                 ethnic = Mode(student_data$ethnic),
                 #initialize new school and student (for random effects)
                 school = "new_school",
                 id = "new_student",
                 #get average age at each measurement point
                 raw_age = avg_measurement_vals$avg_age,
                 #assign the correct fall/spring seasons
                 season = avg_measurement_vals$season,
                 #initialize the COI categories
                 COI_cat_combined = rep(c("Moderate","Very Low / Low","High","Very High"),each=11),
                 #include output (just in case)
                 BMI_t = avg_measurement_vals$BMI_t) %>%
      dplyr::mutate(scaled_age = raw_age - 5)
  }
  else if (nbhd_factor == "ALAN"){
    predictors_df <- 
      data.frame(#get most common values for sex and ethnicity
                 sex = Mode(student_data$sex),
                 ethnic = Mode(student_data$ethnic),
                 #initialize new school and student (for random effects)
                 school = "new_school",
                 id = "new_student",
                 #get average age at each measurement point
                 raw_age = avg_measurement_vals$avg_age,
                 #assign the correct fall/spring seasons
                 season = avg_measurement_vals$season,
                 #get DMSP "categories" (candidate values)
                 DMSP = rep(c(20,40,60),each=11),
                 #include dependent variable (just in case)
                 BMI_t = avg_measurement_vals$BMI_t) %>%
      mutate(scaled_age = raw_age - 5)
  }
  
  return(predictors_df)
}

#make point estimate predictions
   
get_point_predictions <- function(lme4_model, predictors_df){
  # This function generates point estimate predictions and prediction intervals
  #   of a linear mixed effects model.
  #
  # Args:
  #   - lme4_model: a fitted linear mixed effects model
  #   - nbhd_factor: the neighborhood-level factor to be investigated (currently
  #                   only supports "COI" or "ALAN")
  # Returns:
  #   - predicted_vals: the predicted BMI values of a linear mixed effects model,
  #                         plus prediction intervals
  
  # Generate point estimate predictions and prediction intervals
  predicted_vals <- predictInterval(lme4_model, predictors_df,
                                        n.sims = 5000,
                                        seed = 1234)
 
  return(predicted_vals)
}

display_bmi_preds <- function(predictors_df, predicted_vals, nbhd_factor){
  # This function generates point estimate predictions of a linear mixed effects model.
  #
  # Args:
  #   - predictors_df: a dataframe containing the predictors for a linear mixed
  #                    effects model
  #   - predicted_vals: the predicted BMI values of a linear mixed effects model
  #   - nbhd_factor: the neighborhood-level factor to be investigated (currently
  #                   only supports "COI" or "ALAN")
  # Returns:
  #   - store_viz: a plot showing BMI trajectory predictions with confidence intervals
  
  #set color palette manually
  if (nbhd_factor == "COI"){
      coi_colors <- c("Very Low / Low" = "#edf8f8", "Moderate" = "#2CA25F", 
                      "High" = "#217847", "Very High" = "#004d1f")
      coi_colors_fill <- c("Very Low / Low" = "#B2E2E2", "Moderate" = "#66C2A4", 
                           "High" = "#2CA25F", "Very High" = "#006D2C")
  
    store_viz <-
      cbind(predictors_df, predicted_vals) %>%
      #ensure COI categories are a factor variable
      mutate(COI_cat = factor(COI_cat_combined, levels = c("Very Low / Low","Moderate","High","Very High"))) %>%
      ggplot(aes(x = raw_age, y = fit)) +
      #plot prediction intervals
      geom_ribbon(aes(ymin = lwr, ymax = upr, fill = COI_cat), alpha = 0.2,
                  show.legend = FALSE) +
      #plot predicted trajectory lines
      geom_line(aes(col = COI_cat), linewidth = 1.1) +
      theme_minimal() +  
      theme(plot.title = element_text(size = 16, hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5),
            legend.key = element_rect(fill = "lightgray")) +
      scale_color_manual(values = coi_colors) +
      scale_fill_manual(values = coi_colors_fill) +
      labs(y = "Predicted BMI Value", x = "Age (years)",
           col = "COI Category",
           title = "Predicted BMI Trajectories by COI Category",
           subtitle = paste0("Covariate values held constant at sample means, modes."))
  }
  #set color palette manually
  else if (nbhd_factor == "ALAN"){
    alan_colors <- c("20" = "#08306B", "40" = "#6BAED6", "60" = "white")
    alan_colors_fill <- c("20" = "#08306B", "40" = "#6BAED6", "60" = "lightgray")
    
    store_viz <-
      cbind(predictors_df, predicted_vals) %>%
      #ensure ALAN variable is a factor variable
      mutate(DMSP = factor(DMSP)) %>%
      ggplot(aes(x = raw_age, y = fit)) +
      #plot prediction intervals
      geom_ribbon(aes(ymin = lwr, ymax = upr, fill = DMSP), alpha = 0.3,
                  show.legend = FALSE) +
      #plot predicted trajectories
      geom_line(aes(col = DMSP), linewidth = 1.1) +
      theme_minimal() +
      theme(plot.title = element_text(size = 16, hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5),
            legend.key = element_rect(fill = "lightgray")) +
      scale_color_manual(values = alan_colors) +
      scale_fill_manual(values = alan_colors_fill) +
      labs(y = "Predicted BMI Value", x = "Age (years)",
           col = "ALAN Value\n(nanowatt
per steradian\nper square)",
           title = "Predicted BMI Trajectories by ALAN Value",
           subtitle = paste0("Covariate values held constant at sample means, modes."))
    
  }
  
  #ggsave(paste("bmi_traj_", nbhd_factor, ".pdf", sep = ""), store_viz, width = 7.5, height = 3.25)
  
  return(store_viz)
}
 
## _____________________________________________________________________________

## MODEL VALIDATION

## First: set up all models for comparison.

bmi_model_coi_full <- function(input_data){
  ## function to fit "full" COI model
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + c(COI_cat_combined)*c(season)*scaled_age + 
                          I(scaled_age^2) +
                          (1 + scaled_age | school) + (1 + scaled_age | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}



bmi_model_coi_noage2 <- function(input_data){
  ## function to fit "full" COI model without nonlinear age term
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + c(COI_cat_combined)*c(season)*scaled_age + 
                          (1 + scaled_age | school) + (1 + scaled_age | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}

bmi_model_coi_noage2_noageslopes <- function(input_data){
  ## function to fit "full" COI model without nonlinear age term and random slopes
  ##        for raw age
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + c(COI_cat_combined)*c(season)*scaled_age + 
                          (1 | school) + (1 | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}

bmi_model_coi_noageslopes <- function(input_data){
  ## function to fit "full" COI model random slopes for raw age
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + c(COI_cat_combined)*c(season)*scaled_age +
                          I(scaled_age^2) +
                          (1 | school) + (1 | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}

##set up all ALAN models to compare

bmi_model_alan_full <- function(input_data){
  ## function to fit "full" COI model
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + DMSP*c(season)*scaled_age + 
                          I(scaled_age^2) +
                          (1 + scaled_age | school) + (1 + scaled_age | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}


bmi_model_alan_noage2 <- function(input_data){
  ## function to fit "full" COI model without nonlinear age term
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + DMSP*c(season)*scaled_age + 
                          (1 + scaled_age | school) + (1 + scaled_age | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}

bmi_model_alan_noage2_noageslopes <- function(input_data){
  ## function to fit "full" COI model without nonlinear age term and random slopes
  ##        for raw age
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + DMSP*c(season)*scaled_age + 
                          (1 | school) + (1 | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}


bmi_model_alan_noageslopes <- function(input_data){
  ## function to fit "full" COI model random slopes for raw age
  ## Args: input_data (a dataframe of data on which to train the model; must contain 
  ##                   all predictors and the response variable)
  ## Returns: bmi_model_coi (lme4 model)
  
  ## set seed for replicability
  set.seed(101)
  
  ## fit model 
  bmi_model_coi <- lmer(BMI ~ c(sex) + c(ethnic) + DMSP*c(season)*scaled_age +
                          I(scaled_age^2) +
                          (1 | school) + (1 | id),
                        data = input_data, REML = FALSE,
                        control = lmerControl(optimizer ='bobyqa'))
  return(bmi_model_coi)
}

## Second: apply model validation/comparison metrics (including cross-validation using
## RMSE and MAE)

blockCV <- function(school_label, lme4_model, metric){
  ## function that evaluates model performance on unseen data, here all student
  ## observations within a given school.
  ## Args: school_label (string, label of school used as test data)
  ##       lme4_model (function that fits a given lme4 model given input data)
  ##       metric (desired performance metric; currently supports "RMSE" and "MAE")
  ## Returns: test_val (float: the value of the given performance metric of the given 
  ##                           lme4_model on unseen data (the given school))
  
  ## check for correct metric input
  if(!(metric %in% c("RMSE","MAE","RMSE & MAE"))){
    return("ERROR: metric input not recognized. Supported values include 'RMSE' and 'MAE'")}
  
  ## get test (school X only) and train (all but school X) datasets
  test_data <- student_data %>% filter(school == school_label)
  train_data <- student_data %>% filter(school != school_label)
  
  ## train input lme4_model on train_data
  trained_lme4 <- lme4_model(train_data)
  
  ## get predictions
  bmi_cv_predictions <- predict(trained_lme4, test_data, allow.new.levels = TRUE)
  
  ## get prediction errors
  test_pred_resid <- data.frame(BMI_test = test_data$BMI, BMI_pred = bmi_cv_predictions) %>% 
    mutate(resid = BMI_pred - BMI_test)
  
  ## calculate RMSE
  test_rmse <- sqrt(mean(test_pred_resid$resid^2, na.rm = TRUE))
  
  ## calculate MAE
  test_mae <- mean(abs(test_pred_resid$resid), na.rm = TRUE)
  
  ## select desired metric
  if(metric == "RMSE"){test_val <- test_rmse} else if(metric == "MAE"){
    test_val <- test_mae} else if(metric == "RMSE & MAE"){test_val <- c(test_rmse, test_mae)}
  
  ## concatenate and print progress indicator
  cat("School", school_label, "done.\n")
  
  ## return desired metric
  return(test_val)
}
       

apply_blockCV <- function(input_data, input_model){
  ## function to apply blockCV across all values of school labels
  schools_unique <- levels(input_data$school)
  vapply(schools_unique, FUN = blockCV, lme4_model = input_model,
         metric = "RMSE & MAE", FUN.VALUE = numeric(2))
}

# ------------------------------------------------------------------------------------------------------

# END OF FUNCTIONS #