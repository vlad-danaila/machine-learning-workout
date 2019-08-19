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

# Backword elimination

selected_indices = seq(length(data_train))
selected_data_train = data_train[, selected_indices]

regressor = lm(formula = Profit ~ ., data = selected_data_train)
summary = summary(regressor)
p_values = (summary$coefficients)[, 4]
adj_r_squared = summary$adj.r.squared

max_p_value_index = which(p_values == max(p_values))
new_selected_indeices = selected_indices[-max_p_value_index]

