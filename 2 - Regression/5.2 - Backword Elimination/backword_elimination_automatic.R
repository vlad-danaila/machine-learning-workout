data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups - Copy.csv')

# Categorical features
state_levels = levels(data$State)
data$State = factor(data$State, levels = state_levels, labels = seq(length(state_levels)))

# Feature scaling
data[2:4] = scale(data[2:4])

# Test - train split
library('caTools')
is_train = sample.split(data$State, SplitRatio = 0.8)
data_train = subset(data, is_train)
data_test = subset(data, !is_train)

