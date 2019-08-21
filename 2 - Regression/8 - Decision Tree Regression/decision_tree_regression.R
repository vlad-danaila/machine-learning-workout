# Imports
installed.packages('rpart')
library('rpart')

# Data loading
data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
data = data[-1]

# Decision tree
regressor = rpart(formula = Salary ~ ., data = data, control = rpart.control(minsplit = 1))

# Making predicitons
x_grid = data.frame(Level = seq(min(data$Level), max(data$Level), 0.001))
pred = predict(regressor, x_grid)

# Plot
ggplot() + 
  geom_point(aes(x = data$Level, y = data$Salary), color = 'red') +
  geom_line(aes(x = x_grid$Level, y = pred), color = 'blue') + 
  ggtitle('Decision tree') + 
  xlab('Career salary') + 
  ylab('Salary')
  
