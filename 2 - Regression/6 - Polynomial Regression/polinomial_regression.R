data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
data = data[-1]

polynomial_degree = 5

# Data preparation
add_polynomial_features = function(dataframe, features, degree = 2, feat_name = 'feat') {
  for (i in seq(2, degree)) {
    dataframe[paste(feat_name, '_pow_', i)] = features ^ i
  }
  dataframe
}
data = add_polynomial_features(data, data$Level, degree = polynomial_degree, feat_name = 'Level')

# Polinomial regression
regressor = lm(formula = Salary ~ ., data = data)

# Predictions 
x = seq(min(data$Level), max(data$Level), by = 0.001)
x_pred = add_polynomial_features(data.frame(Level = x), x, degree = polynomial_degree, feat_name = 'Level')
pred = predict(regressor, newdata = x_pred)

# Plot
library('ggplot2')
ggplot() +
  geom_point(aes(x = data$Level, y = data$Salary), color = 'red') +
  geom_line(aes(x = x, y = pred), color = 'blue')