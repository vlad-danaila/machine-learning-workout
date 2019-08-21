# Imports
# For decision tree
# installed.packages('rpart')
library('rpart')
# For random forest
#install.packages('randomForest')
library(randomForest)

# Data loading
data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
data = data[-1]

# Decision tree
regressor_tree = rpart(formula = Salary ~ ., data = data, control = rpart.control(minsplit = 1))

# Random forest
regressor_forest = randomForest(x = data[-2], y = data$Salary, ntree = 500)

# Making predicitons
x_grid = data.frame(Level = seq(min(data$Level), max(data$Level), 0.001))
pred_tree = predict(regressor_tree, x_grid)
pred_forest = predict(regressor_forest, x_grid)

# Plot
ggplot() + 
  geom_point(aes(x = data$Level, y = data$Salary), color = 'red') +
  geom_line(aes(x = x_grid$Level, y = pred_tree), color = 'yellow') + 
  geom_line(aes(x = x_grid$Level, y = pred_forest), color = 'orange') + 
  ggtitle('Decision tree') + 
  xlab('Career salary') + 
  ylab('Salary')
  
