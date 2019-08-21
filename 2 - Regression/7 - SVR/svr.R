#install.packages('e1071')
library(e1071)
library(ggplot2)

# Data loading
data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
data = data[-1]

# Feature scalling
data_scaled = scale(data)

# SVR
regressor = svm(formula = Salary ~ ., data = data_scaled)

# Make predictions
x = data.frame(Level = seq(min(data$Level), max(data$Level), 0.1))
x_scaled = scale(x)
pred_scaled = predict(regressor, newdata = x_scaled)

# Unscale predictions
mean = attr(data_scaled,"scaled:center")['Salary']
variance = attr(data_scaled,"scaled:scale")['Salary']
pred = pred_scaled * variance + mean 

# Plot
ggplot() + 
  geom_point(aes(x = data$Level, y = data$Salary), color = 'red') + 
  geom_line(aes(x = x$Level, y = pred), color = 'blue') + 
  ggtitle('SVR') + 
  xlab('Career level') + 
  ylab('Salary')
