data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')

# Train test split
library(caTools)
is_split = sample.split(data$YearsExperience, SplitRatio = 0.7)
train = subset(data, is_split)
test = subset(data, !is_split)

# Linear regression
regressor = lm(Salary ~ ., data = train)
pred_train = predict(regressor, train)
pred_test = predict(regressor, test)

library(ggplot2)
ggplot() +
  geom_point(aes(test$YearsExperience, test$Salary)) + 
  geom_line(aes(test$YearsExperience, pred_test), colour = 'red') + 
  ggtitle('Linear regression') + 
  xlab('Years') + 
  ylab('Salary')