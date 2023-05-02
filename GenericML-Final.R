library(GenericML)
library(grf)
library(glmnet)
library(haven)
library(magrittr)
library(ggplot2)
library(dplyr)

# --- Data Loading, Subletting
df <- read.csv('/Users/jakecosgrove/Documents/Thesis/thesis-code/data/Final-Data-2023-04-11.csv')

# Select treatment and outcome variables you want to use
df$Treatment <- df$wb24
df$Outcome <- df$weak_share_weak_envy_strong
# df$Treatment <- ifelse(df$wb23==1 & df$trigger==1, 1, 0)
# df$Treatment <- df$treatment
# df$Outcome <- df$weak_share

# subset the data to include outcome, treatment, and covariates of interest
data <- subset(df, select = c("Outcome", "Treatment",
                              "q2_age", "Male", 'USA', 'India', 'Kenya', 'Mexico',
                              "q14_socioeconomic_status", "q15_income_consistency", "satisfaction_pca", "stress_factor", 
                              "cognition", "q84b_friends", "hungry_4hr", "trust_pca", "outside_wetbub_c", "satisfaction_pca"))
# drop any rows that contain missing observations
model.data <- data[complete.cases(data),]

# Select outcome, treatment, and covariates seperately
Z <- select(model.data, -c("Outcome","Treatment"))
Y <- model.data$Outcome
D <- model.data$Treatment
Z <- as.matrix(Z)
Y <- as.numeric(Y)
D <- as.numeric(D)

# --- Specifying Parameters for Generic ML 
# Tell it to use elastic net (=0.5) or lasso (=1.0):
# specify the learner of the propensity score (non-penalized logistic regression here). Propensity scores can also directly be supplied.
learner_propensity_score <- "mlr3::lrn('glmnet', lambda = 0, alpha = 1)"
# specify the considered learners of the BCA and the CATE (here: lasso, random forest, and SVM)
learners_GenericML <- c("lasso", "mlr3::lrn('ranger', num.trees = 100)","mlr3::lrn('svm')")
# specify the data that shall be used for the CLAN
# here, we use all variables of Z and uniformly distributed random noise
Z_CLAN <- Z
# Below are parameters as followed from the generic ML documentation
X1_BLP   <- setup_X1()
X1_GATES <- setup_X1()
vcov_BLP   <- setup_vcov()
vcov_GATES <- setup_vcov()
diff_GATES <- setup_diff(subtract_from = "most",
                         subtracted = 1:3)
diff_CLAN  <- setup_diff(subtract_from = "most",
                         subtracted = 1:3)
quantile_cutoffs <- c(0.2, 0.4, 0.6, 0.8)
num_splits <- 100
HT <- FALSE
equal_variances_CLAN <- FALSE
prop_aux <- 0.5
stratify <- setup_stratify()
store_splits   <- TRUE
store_learners <- FALSE
parallel  <- FALSE
num_cores <- 8      
seed      <- 79441831
significance_level <- 0.05
min_variation <- 1e-05

# --- Running the Generic ML model
genML <- GenericML(Z = Z, D = D, Y = Y, 
                   learner_propensity_score = learner_propensity_score,
                   learners_GenericML = "lasso", #learners_GenericML,
                   num_splits = num_splits,
                   Z_CLAN = Z_CLAN,
                   HT = HT,
                   X1_BLP = X1_BLP,
                   X1_GATES = X1_GATES,
                   vcov_BLP = vcov_BLP,
                   vcov_GATES = vcov_GATES,
                   quantile_cutoffs = quantile_cutoffs,
                   diff_GATES = diff_GATES,
                   diff_CLAN = diff_CLAN,
                   equal_variances_CLAN = equal_variances_CLAN,
                   prop_aux = prop_aux,
                   stratify = stratify,
                   significance_level = significance_level,
                   min_variation = min_variation,
                   parallel = parallel,
                   num_cores = num_cores,
                   seed = seed,
                   store_splits = store_splits,
                   store_learners = store_learners)

# --- looking into model results
summary(genML)

# Now getting Best Linear Predictor and Significance of HTEs:
results_BLP <- get_BLP(genML, plot = TRUE)
results_BLP # print & plot method
plot(results_BLP) # plot method

# Obtain Sorted Group Average Treatment Effects (GATES):
results_GATES <- get_GATES(genML, plot = TRUE)
results_GATES

