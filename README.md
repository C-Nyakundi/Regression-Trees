# Regression Trees 

Predicting weight using height and gender in regression tree models

# -------------------------------Predicting weights using height & Gender------------------------------------------------


##Introduction

This project aims to predict individuals' weights based on their heights and genders. Understanding the relationship between these variables can provide valuable insights into health and wellness. We'll explore two approaches: linear regression and regression tree models

# -------------------------------Steps to follow------------------------------------------------

## 1- Importing the CSV dataset
## 2- Plotting the data (Checking for normality)
## 3- Defining the dependent & independent variables (Intially we will not consider using gender)
## 4- Splitting data into train and test datasets 
## 5- Creating a linear regression model
## 6- Checking the performance of the model
## 7- Adding gender to the model. 

# Loading libraries 
library(pacman)
pacman::p_load(tidyverse, broom, finalfit, rpart, rpart.plot, Metrics)
#broom- used to get variable(s) & specific values in a nice tibble
#finalfit- creation of final tables and reports summarizing statistical models. 
#Metrics- for computing various evaluation metrics commonly used in machine learning and statistical analysis
#rpart- for building classification and regression trees (CART). It implements the Recursive Partitioning and Regression Trees algorithm, which recursively splits the data into subsets based on the values of input variables, creating a decision tree.

# Importing dataset
data <- read.csv("gender-height-weight.csv")

# EDA
class(data)
#(dataframe)
names(data) # Variable names
glimpse(data) 
sapply(data, class) # 1 (Gender) string var & weight & height (numericals)

# Plotting the data 
data %>%
  ggplot(aes(y=Height..cm., x=Weight..kg., colour=Gender)) +
  geom_point(alpha=0.2) +
  geom_smooth(method = "lm", se=F)

# Splitting dataset into training & testing 
training <- data[seq(1, nrow(data),2),]
testdata <- data[seq(2, nrow(data),2),]

# Extract the features weight is in kg, height is in cm and gender
training_weight <- training[,5]
training_height <- training[,4]
training_sex <- training[,1]

# Set the training features
test_weight <- testdata[,5]
test_height <- testdata[,4]

# Creating a linear regression model
fit <- lm(training_weight~training_height)

# Plotting the regression model 
plot(training_height, training_weight, col='blue',xlab = 'height (cm)', ylab = 'weight (kg)')
abline(fit, col='red')
# This code generates a scatter plot of data points representing training_height on the x-axis and training_weight on the y-axis. 
# The points are colored blue, and the x-axis is labeled 'height (cm)' while the y-axis is labeled 'weight (kg)'. 
# Additionally, it overlays a regression line onto the plot, which is drawn based on the linear regression model fit stored in the variable 'fit'.The regression line is colored red.



# Print fit model to see the coeefficients 
pred_weight <- 1.372 * test_height -158.101

# This code calculates predicted weights based on the formula pred_weight = 1.372 × test_height − 158.101
# pred_weight=1.372×test_height−158.101. It uses the test_height values to predict corresponding weights, assuming a linear relationship between height and weight with coefficients obtained from a linear regression model or some other source.

# Calculating the mean error 
er <- mean(pred_weight - test_weight)
# The mean error (er) between the predicted weights (pred_weight) and the actual weights (test_weight). 
# It subtracts each predicted weight from its corresponding actual weight, calculates the mean of these differences, and assigns the result to the variable 'er'. 
# This provides an indication of the average deviation between the predicted and actual weights.


# Calculate the root mean squared error
rmse <- sqrt(sum((pred_weight - test_weight)^2)/length(pred_weight))
# measure of the average magnitude of the prediction errors.


fit %>% glance()

print(summary(fit))


# --------------- Running a regression tree on the weights/heights data------------------------------

# Model 2: Building a regression tree
set.seed(3846)

ind <- sample(2, nrow(data), replace = T, prob = c(0.5, 0.5))
# The provided code generates a vector ind of random integers representing indices. 
# Each integer is randomly sampled from the set {1, 2} with replacement, where the probability of selecting each value is equal (0.5 for each). 
# The length of the vector ind is equal to the number of rows in the dataset data.

train <- data[ind == 1,]
test <- data[ind == 2,]
# 
# The code separates the dataset data into two subsets: a training set (train) and a test set (test). 
# It does so by indexing the dataset using the vector ind.
# 
# For the training set, it selects rows where the corresponding element in the ind vector is equal to 1.
# For the test set, it selects rows where the corresponding element in the ind vector is equal to 2.
# This partitioning is based on the random sampling previously performed using the sample() function.


# Tree Classification 
tree <- rpart(Weight..kg. ~ Height..cm.+Gender, data = train, method = "anova")
# The code fits a regression tree model (rpart) to predict the weight in kilograms (Weight..kg.) based on the height in centimeters (Height..cm.) and gender (Gender) variables. 
# The method used for splitting the tree is "anova", which indicates that the model will use analysis of variance (ANOVA) to determine the best splits for the predictor variables. 
# The model is trained on the train dataset.
rpart.plot(tree, extra = 1)
# The code generates a visual representation of the regression tree model (tree) using the rpart.plot function. 
# The extra = 1 argument adds additional information to the plot, such as the number of observations in each node. 
# This visualization helps in interpreting the decision-making process of the regression tree model and understanding how predictor variables are used to partition the data

printcp(tree)
# The printcp() function is used to display the complexity parameter (CP) table of the regression tree model tree. 
# This table provides information about the complexity parameter values and corresponding measures of model complexity, such as the number of splits, relative error, and cross-validated error, for each value of the CP. 
# It helps in selecting an appropriate CP value for pruning the tree and avoiding overfitting.

plotcp(tree)
![plot] (C:/Users/ELITEBOOK/OneDrive/Pictures/Documents/Data Science with R/Regression-Trees-/Regression analysis/optimal value of the complexity parameter for pruning the tree.png)
# The plotcp() function is used to visualize the cross-validation results for different complexity parameter (CP) values in the regression tree model (tree).
This plot helps in identifying the optimal value of the complexity parameter for pruning the tree. The x-axis represents the CP values, while the y-axis represents the cross-validated error.
The plot typically shows a U-shaped curve, and the optimal CP value is often chosen as the one that minimizes the cross-validated error.

#Generate prediction on a test set 
pred <- predict(object = tree,  # Model object
                newdata = test) # test data set

The code generates predictions (pred) for the test dataset using the regression tree model (tree
