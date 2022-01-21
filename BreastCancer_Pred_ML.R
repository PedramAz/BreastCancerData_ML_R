# Breats Cancer diagnosis prediction using Machine Learning 
bc <- read.csv("C:/Users/azimz/Desktop/BreastCancer_Pred_ML/data.csv") 

# Data exploration
dim(bc)
names(bc)
head(bc, 2)
str(bc)

# Count the number of NAs in the whole dataframe 
sum(is.na(bc))

# remove X column because it is all NAs
library(dplyr)
bc <- select(bc, -X)
names(bc)

# Checkout the main dependent variable 
table(bc$diagnosis)
typeof(bc$diagnosis)

bc$diagnosis <- recode(bc$diagnosis, "B" = 1, "M" = 2)

# Visual exploration 
par(mfrow = c(2,2))
plot(bc$diagnosis, bc$radius_mean)
plot(bc$diagnosis, bc$texture_mean)
plot(bc$diagnosis, bc$perimeter_mean)
plot(bc$diagnosis, bc$area_mean)

# Create a new df without diagnosis column 
bc_n <- select(bc, -diagnosis)
str(bc_n)

# Check variable correlations 
library(corrplot)
par(mfrow = c(1, 1))
corrplot(cor(bc_n))


# Data preprocessing 
# Data partitioning -> train and test 
library(caret)
# sort the dataset 
bc <- bc[order(bc$diagnosis),]
set.seed(5432)

# Creating indexes
trainIndices <- createDataPartition(bc$diagnosis, p = 0.75, list = FALSE)

# training data
trainBC <- bc[trainIndices, ]
nrow(trainBC)
# test data
testBC <- bc[-trainIndices, ]
nrow(testBC)

# Logistic regression 
formula = diagnosis ~ perimeter_mean + concavity_mean + area_mean + smoothness_mean + texture_mean + compactness_mean

lr <- glm(formula, data = trainBC)
summary(lr)

# Calculate predictions on testing dataset 
pred_lr <- predict(lr, newdata = testBC)
summary(pred_lr)
plot(pred_lr, testBC$diagnosis)

# define a function to calculate sum of squares 
sumOfSquares <- function(x){
  return(sum(x^2))
}

# calculate residuals 
diff_lr <- pred_lr - testBC$diagnosis
# use residuals to calculate sum of squares as an accuracy measure 
sumOfSquares(diff_lr)

# Diagnostic plots 
par(mfrow = c(2, 2))
plot(lr)


# Create a linar regression model using the same formula
lin_reg <- lm(formula, data = trainBC)
summary(lin_reg)
plot(resid(lin_reg))
# predictions on test data 
pred_lin <- predict(lin_reg, newdata = testBC)

diff_lin <- pred_lin - testBC$diagnosis

sumOfSquares(diff_lin)

# Calculate relative importance
# lmg: the coefficient of the variable from the model
# last: looks at what the effect of adding this variable into the model
# First: looks at the variable as if none of the other variables were present in the model
# Pratt: product of the standard coefficient and the correlation
library(relaimpo)
importance <- calc.relimp(lin_reg, type = c("pratt", "lmg", "last", "first"), rela = FALSE)
importance
plot(importance)

bootresults <- boot.relimp(lin_reg, b = 1000)
ci <- booteval.relimp(bootresults, norank = T)
plot(ci)

# Variable selection methods 
# 1- All possible subsets 

library(leaps)
all_poss <- regsubsets(formula, data = bc, nbest = 1, nvmax = 7)
info <- summary(all_poss)
cbind(info$which, round(cbind(rsq = info$rsq, adjr2 = info$adjr2, cp = info$cp, bic = info$bic, rss = info$rss), 3))

# 2- Backward elimination
# This model works based on AIC
# AIC is a math method for evaluating how well a model fits the data 
library(MASS)
null <- lm(diagnosis ~ 1, data = bc)
full <- lm(formula, data = bc)
stepAIC(full, scope = list(lower = null, upper = full), data = bc, direction = 'backward')

# 2.1- Stepwise 
stepAIC(full, scope = list(lower = null, upper = full), data = bc, direction = 'both')











