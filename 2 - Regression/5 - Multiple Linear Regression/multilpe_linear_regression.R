data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# Feature scaling
for (i in 1:3) {
  data[,i] = scale(data[, i])
}

# Categorical data
state_levels = levels(data$State)
data$State = factor(data$State, levels = state_levels, labels = seq(length(state_levels)))

# Test - train split
library('caTools')
is_train = sample.split(data$Profit, SplitRatio = 0.7)
data_train = subset(data, is_train)
data_test = subset(data, !is_train)

# Regression
regressor = lm(formula = Profit ~ ., data = data_train)
pred_test = predict(regressor, data_test)