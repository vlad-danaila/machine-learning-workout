data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Homework_Solutions/50_Startups.csv')

# Categorical features
state_levels = levels(data$State)
data$State = factor(data$State, levels = state_levels, labels = seq(length(state_levels)))

# Feature scaling
data[, 1:3] = scale(data[, 1:3])

# Train - test split
library(caTools)
is_training = sample.split(data$Profit, SplitRatio = 0.8)
data_train = subset(data, is_training)
data_test = subset(data, !is_training)

# Regression
regressor = lm(formula = Profit ~ ., data = data_train)

# Backwords elimination, step 1
summary(regressor)

# Backwords elimination, step 2
regressor = lm(
  formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
  data = data_train
)
summary(regressor)

# Backwords elimination, step 3
regressor = lm(
  formula = Profit ~ R.D.Spend + Marketing.Spend, 
  data = data_train
)
summary(regressor)

# Backwords elimination, step 4
regressor = lm(
  formula = Profit ~ R.D.Spend, 
  data = data_train
)
summary(regressor)
