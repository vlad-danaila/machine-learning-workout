#install.packages('class')
library(class)
#install.packages('ElemStatLearn')
library(ElemStatLearn)
library(caTools)

# Load data
data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 15 - K-Nearest Neighbors (K-NN)/Social_Network_Ads.csv')
data = data[3:5]
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

data[1:2] = scale(data[1:2])

# Train - test split
is_train = sample.split(data$Purchased, SplitRatio = 0.7)
data_train = subset(data, is_train)
data_test = subset(data, !is_train)

# KNN classification on test set
pred_test = knn(
  train = data_train[-3], test = data_test[-3], cl = data_train$Purchased, k = 5)

# Confusion matrix
cm = table(data_test$Purchased, pred_test)
accuracy = sum(diag(cm)) / sum(cm)
cat('Accuracy is', accuracy)

# Make grid for plotting
make_grid = function(v, steps = 100) {
  v_limit_min = min(v) - 1
  v_limit_max = max(v) + 1
  seq(v_limit_min, v_limit_max, by = (v_limit_max - v_limit_min) / steps)
}
grid_x_0 = make_grid(data_test$Age)
grid_x_1 = make_grid(data_test$EstimatedSalary)
grid_x_0_1 = expand.grid(grid_x_0, grid_x_1)
colnames(grid_x_0_1) = colnames(data[-3])
grid_pred = knn(
  train = data_train[-3], test = grid_x_0_1, cl = data_train$Purchased, k = 15)

# Display plot
plot(
  data_test$Age, 
  data_test$EstimatedSalary,
  main = 'KNN',
  xlab = 'Age', 
  ylab = 'Sallary',
  xlim = range(grid_x_0),
  ylim = range(grid_x_1))
points(grid_x_0_1[, 1], grid_x_0_1[, 2], pch = 16, col = ifelse(grid_pred == 1, 'tomato', 'springgreen'), alpha = 0.2)
points(data_test$Age, data_test$EstimatedSalary, pch = 16, col = ifelse(data_test$Purchased, 'darkred', 'darkgreen'))
