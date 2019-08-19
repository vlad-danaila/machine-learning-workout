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

# Backword elimination - Not working properly
# The problem is with factors being unpacked in the model but not whem selecting the data
# Wold need to create a one hot encoding as in python
"browser()
selected_indices = seq(length(data_train))
for (i in seq(length(data_train))) {
  selected_data_train = data_train[, selected_indices]
  regressor = lm(formula = Profit ~ ., data = selected_data_train)
  summary = summary(regressor)
  p_values = (summary$coefficients)[, 4]
  adj_r_squared = summary$adj.r.squared
  max_p_value = max(p_values)
  if (max_p_value > 0.05) {
    max_p_value_index = which(p_values == max_p_value)
    new_selected_indeices = selected_indices[-max_p_value_index]
    newModel = lm(formula = Profit ~ ., data = data_train[, new_selected_indeices])
    new_adj_r_squared = summary(newModel)$adj.r.squared
    if (adj_r_squared < new_adj_r_squared) {
      regressor = newModel
      selected_indices = new_selected_indeices
    } 
  } else {  
    break
  }
}
"

