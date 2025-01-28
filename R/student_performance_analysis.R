# ------------------------------------------------------------------------------
# Script Name: Student Performance Analysis
# Description: This module contains the R code for the reported analysis.
# Author: Carlton Berthold
# Date: 2024-12-17
# 
# This script relies on JAGS (Just Another Gibbs Sampler) as the MCMC sampler
# for the 'BayesianMediationA' package. Ensure that JAGS is installed on your
# system before running this script. Also ensure that the 'BayesianMediationA'
# package and its dependencies are installed correctly, and that your JAGS
# install is linked to those packages.
# 
# Dependency:
# - JAGS: https://mcmc-jags.sourceforge.io/
#
# Installation instructions for JAGS:
# - On Windows: Download and install from the official website.
# - On macOS: Install using Homebrew with the command `brew install jags`.
# - On Linux: Install via a package manager, e.g., sudo apt-get install jags`.
# ------------------------------------------------------------------------------


# Load necessary libraries -----------------------------------------------------
library(brms)
library(BayesianMediationA)
library(car)
library(dplyr)
library(ggplot2)
library(loo)
library(Metrics)
library(mice)
library(recipes)


# Set a seed to ensure reproducible randomness ---------------------------------
seed = 480
set.seed(seed)


# Read in data and prepare factor columns --------------------------------------
raw_data = read.table("../Data/StudentPerformanceFactors.csv", sep=',',
                      header = T, stringsAsFactors = T)


# Establish preprocessing pipeline for current and new data --------------------

#' Apply MICE imputation to a dataframe.
#' 
#' @param data Input dataframe.
#' @returns Dataframe with missing values imputed and replaced.
impute_missing = function(data) {
  data[data == ''] = NA
  dry_mice = mice(data, maxit = 0)
  pred = dry_mice$predictorMatrix
  pred[, "Exam_Score"] = 0 # Exclude outcome variable to reduce colinearity.
  imputed_temp = mice(data, m = 5, method = 'pmm', predictorMatrix = pred,
                      maxit = 500, ridge = 0.0001, threshold = 1.1, seed = seed)
  imputed_data = mice::complete(imputed_temp, 1)
  # Drop unused levels for all factor columns to ensure proper encoding
  imputed_data = mutate(data, across(where(is.factor), droplevels))
  return(imputed_data)
}

#' Dummy code a data frame.
#' 
#' The reference level for each factor is assigned based on the lowest internal
#' integer value for that level. Dummy coded factor columns are grouped together
#' for a given factor level and placed to the right of any continuous columns
#' in the data frame.
#' 
#' @param data Data frame to be converted.
#' @returns Data frame with factor columns dummy coded and numeric columns
#' retained.
dummy_code_data = function(data) {
  rec = recipe(~ ., data = data)
  rec = step_dummy(rec, all_nominal_predictors(), -all_outcomes())
  baked_data = bake(prep(rec), new_data = NULL)
  return(baked_data)
}

#' Impute missing values and return original and dummy coded data frames.
#' 
#' @param data Data frame to be preprocessed.
#' @return List of original data frame and dummy coded data after imputation.
preprocess = function(data) {
  imputed_data = impute_missing(data)
  imputed_data = na.omit(imputed_data) # Remove any NA values missed by MICE.
  # It is necessary to dummy code data for the mediation analysis.
  dummy_coded_data = dummy_code_data(imputed_data)
  return(list(imputed_data, dummy_coded_data))
}


# Split data into train and test sets and conduct separate preprocessing -------
train_data = raw_data[1:5000,]
test_data = raw_data[5001:6607,]

train_preprocessed = preprocess(train_data)
exam_data = train_preprocessed[[1]]
dummy_exam_data = train_preprocessed[[2]]

test_preprocessed = preprocess(test_data)
test_data = test_preprocessed[[1]]
dummy_test_data = test_preprocessed[[2]]


# Conduct Bayesian mediation analysis for mediating influences of income -------
outcome = exam_data[, 20] # Exam score.
exposure = exam_data[, 11] # Family income.
mediators = exam_data[, c(4, 10, 12:13)]
outcome_covariates = dummy_exam_data[, -c(5, 7, 10:11, 16:20)]
mediator_covariates = dummy_exam_data[, 24:25]
mediation_output = bma.bx.cy(pred = exposure, m = mediators, y = outcome,
                             cova = outcome_covariates,
                             mcov = mediator_covariates,
                             n.iter=5000, n.burnin = 1000, n.chains = 1)
summary(mediation_output)


# Initialize prior objects for normal and R2D2 brms fits -----------------------
normal_prior = prior(gamma(1, 1), class = sigma) +
  prior(normal(0, 5), class = Intercept) +
  prior(normal(0, 5), class = b)

R2D2_prior = prior(gamma(1, 1), class = sigma) +
  prior(normal(0, 5), class = Intercept) +
  prior(R2D2(mean_R2 = 0.8, prec_R2 = 10, cons_D2 = 0.6, main = T), class = b)


# Initialize formulas for all predictor and partial models ---------------------
full_formula = Exam_Score ~ Gender + Distance_from_Home + 
  Parental_Education_Level + Learning_Disabilities + Physical_Activity +
  Peer_Influence + School_Type + Teacher_Quality + Family_Income +
  Tutoring_Sessions + Internet_Access + Motivation_Level + Previous_Scores + 
  Sleep_Hours + Extracurricular_Activities + Access_to_Resources +
  Parental_Involvement + Attendance + Hours_Studied

hypothesized_relevant_formula = Exam_Score ~ Family_Income + Hours_Studied +
  Attendance + Access_to_Resources + Sleep_Hours + Previous_Scores +
  Tutoring_Sessions + Teacher_Quality + Learning_Disabilities + Gender


# Fit naive full model using normal prior --------------------------------------
naive_fit = brm(
  formula = full_formula,
  data = exam_data,
  prior = normal_prior,
  warmup = 1000,
  iter = 11000,
  chains = 4,
  cores = 4, 
  thin = 1
)


# Fit model using hypothesized major predictors and normal prior ---------------
hypothesized_fit = brm(
  formula = hypothesized_relevant_formula,
  data = exam_data,
  prior = normal_prior,
  warmup = 1000,
  iter = 11000,
  chains = 4,
  cores = 4, 
  thin = 1
)


# Fit full model using R2D2 priors for automatic relevance determination -------
R2D2_fit = brm(
  formula = full_formula,
  data = exam_data,
  prior = R2D2_prior,
  warmup = 3000,
  iter = 13000,
  chains = 4,
  cores = 4, 
  thin = 1,
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)


# Save fitted models as RDS files ----------------------------------------------
saveRDS(mediation_output, "mediation_output.rds")
saveRDS(naive_fit, "naive_fit.rds")
saveRDS(hypothesized_fit, "hypothesized_fit.rds")
saveRDS(R2D2_fit, "R2D2_fit.rds")


# Summarize and plot models to check convergence -------------------------------
summary(naive_fit)
summary(hypothesized_fit)
summary(R2D2_fit)

plot(naive_fit, ask = F)
plot(hypothesized_fit, ask = F)
plot(R2D2_fit, ask = F)


# Graphical checks of generative performance -----------------------------------
pp_check(naive_fit) + theme_bw()
pp_check(hypothesized_fit) + theme_bw()
pp_check(R2D2_fit) + theme_bw()


# Assess predictive performance on test set and calculate metrics --------------
naive_test_pred = predict(naive_fit, newdata = test_data)[, 1]
naive_MAE = mae(test_data$Exam_Score, naive_test_pred)
naive_RMSE = rmse(test_data$Exam_Score, naive_test_pred)
naive_R2 = cor(test_data$Exam_Score, naive_test_pred)^2

hypothesized_test_pred = predict(hypothesized_fit, newdata = test_data)[, 1]
hypothesized_MAE = mae(test_data$Exam_Score, hypothesized_test_pred)
hypothesized_RMSE = rmse(test_data$Exam_Score, hypothesized_test_pred)
hypothesized_R2 = cor(test_data$Exam_Score, hypothesized_test_pred)^2

R2D2_test_pred = predict(R2D2_fit, newdata = test_data)[, 1]
R2D2_MAE = mae(test_data$Exam_Score, R2D2_test_pred)
R2D2_RMSE = rmse(test_data$Exam_Score, R2D2_test_pred)
R2D2_R2 = cor(test_data$Exam_Score, R2D2_test_pred)^2

predictive_summary = data.frame(
  Fit = c("Naive_Full", "Hypothesized_Relevant", "R2D2_Prior"),
  MAE = c(naive_MAE, hypothesized_MAE, R2D2_MAE),
  RMSE = c(naive_RMSE, hypothesized_RMSE, R2D2_RMSE),
  R2 = c(naive_R2, hypothesized_R2, R2D2_R2)
)


# Check assumptions and criticize models ---------------------------------------
# Analyze naive fit.
naive_train_pred = predict(naive_fit)[, 1]
naive_residuals = exam_data$Exam_Score - naive_train_pred
# Residuals vs Fitted values plot (training data)
ggplot(data = NULL, aes(x = naive_train_pred, y = naive_residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Naive Fit Residuals vs Fitted Values (Training Data)",
       x = "Fitted Values",
       y = "Residuals") +
  theme_bw()
# Q-Q plot of residuals (training data)
ggplot(data = NULL, aes(sample = naive_residuals)) +
  geom_qq() +
  geom_qq_line(color = "red") +
  labs(title = "Naive Fit Q-Q Plot of Residuals (Training Data)",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_bw()

# Analyze hypothesized fit.
hypo_train_pred = predict(hypothesized_fit)[, 1]
hypo_residuals = exam_data$Exam_Score - hypo_train_pred
# Residuals vs Fitted values plot (training data)
ggplot(data = NULL, aes(x = hypo_train_pred, y = hypo_residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Hypothesized Fit Residuals vs Fitted Values (Training Data)",
       x = "Fitted Values",
       y = "Residuals") +
  theme_bw()
# Q-Q plot of residuals (training data)
ggplot(data = NULL, aes(sample = hypo_residuals)) +
  geom_qq() +
  geom_qq_line(color = "red") +
  labs(title = "Hypothesized Fit Q-Q Plot of Residuals (Training Data)",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_bw()

# Analyze R2D2 fit.
R2D2_train_pred = predict(R2D2_fit)[, 1]
R2D2_residuals = exam_data$Exam_Score - R2D2_train_pred
# Residuals vs Fitted values plot (training data)
ggplot(data = NULL, aes(x = R2D2_train_pred, y = R2D2_residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "R2D2 Fit Residuals vs Fitted Values (Training Data)",
       x = "Fitted Values",
       y = "Residuals") +
  theme_bw()
# Q-Q plot of residuals (training data)
ggplot(data = NULL, aes(sample = R2D2_residuals)) +
  geom_qq() +
  geom_qq_line(color = "red") +
  labs(title = "R2D2 Fit Q-Q Plot of Residuals (Training Data)",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_bw()


# Model comparison of fit using ELPD loo ---------------------------------------
loo_results = loo_compare(loo(naive_fit), loo(hypothesized_fit), loo(R2D2_fit))


# Model comparison of fit using WAIC -------------------------------------------
naive_waic = add_criterion(naive_fit, 'waic')$criteria$waic
hypothesized_waic = add_criterion(hypothesized_fit, 'waic')$criteria$waic
R2D2_waic = add_criterion(R2D2_fit, 'waic')$criteria$waic


# Export summary of best fit model----------------------------------------------
#' Export a brms summary object to a CSV.
#' 
#' @param summary_obj Summary object to export.
#' @param file_name Name of the exported file.
export_summary = function(summary_obj, file_name) {
  # Extract fixed effects.
  fixed_effects = as.data.frame(summary_obj$fixed)
  fixed_effects = cbind(Effect = rownames(fixed_effects), fixed_effects)
  rownames(fixed_effects) = NULL
  # Extract and combine random effects if present.
  if ("random" %in% names(summary_obj)) {
    random_effects = do.call(rbind, lapply(summary_obj$random, function(x) {
      x = as.data.frame(x)
      x = cbind(Effect = rownames(x), x)
      rownames(x) = NULL
      return(x)
    }))
    combined_summary = rbind(fixed_effects, random_effects)
  } else {
    combined_summary = fixed_effects
  }
  # Write to CSV.
  write.csv(combined_summary, file_name, row.names = FALSE)
}

export_summary(summary(R2D2_fit), "R2D2_Summary.csv")
export_summary(summary(naive_fit), "naive_Summary.csv")


# Generate conditional effects plots for most important predictors -------------
effects = c("Access_to_Resources", "Parental_Involvement", "Motivation_Level",
            "Family_Income", "Teacher_Quality", "Learning_Disabilities",
            "Internet_Access", "Distance_from_Home", "Peer_Influence")
conditional_effects = conditional_effects(naive_fit, effects = effects)
all_plots = plot(conditional_effects, plot = FALSE, ask = FALSE)

all_plots$Access_to_Resources +
  theme_bw() + 
  labs(x = "Access to Resources", y = "Exam Score")

all_plots$Parental_Involvement +
  theme_bw() + 
  labs(x = "Parental Involvement", y = "Exam Score")

all_plots$Motivation_Level +
  theme_bw() + 
  labs(x = "Motivation Level", y = "Exam Score")

all_plots$Family_Income +
  theme_bw() + 
  labs(x = "Family Income", y = "Exam Score")

all_plots$Teacher_Quality +
  theme_bw() + 
  labs(x = "Teacher Quality", y = "Exam Score")

all_plots$Learning_Disabilities +
  theme_bw() + 
  labs(x = "Learning Disabilities", y = "Exam Score")

all_plots$Internet_Access +
  theme_bw() + 
  labs(x = "Internet Access", y = "Exam Score")

all_plots$Distance_from_Home +
  theme_bw() + 
  labs(x = "Parental Involvement", y = "Exam Score")

all_plots$Peer_Influence +
  theme_bw() + 
  labs(x = "Peer Influence", y = "Exam Score")
