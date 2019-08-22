#install.packages('class')
library(class)
#install.packages('ElemStatLearn')
library(ElemStatLearn)
library(caTools)

# Load data
data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 15 - K-Nearest Neighbors (K-NN)/Social_Network_Ads.csv')
data = data[3:5]

# Train - test split
is_train = sample.split(data$Purchased, SplitRatio = 0.7)
data_train = subset(data, is_train)
data_test = subset(data, !is_train)

# Feature scaling
means = apply(data_train[1:2], 2, mean)
variances = apply(data_train[1:2], 2, var)
scale_matrix = function(matrix, means, variances) {
 matrix = matrix - means
 matrix = matrix / variances
 matrix
}
data_train_scaled = data.frame(data_train)
data_train_scaled[1:2] = scale_matrix(data_train_scaled[1:2], means, variances)
data_test_scaled = data.frame(data_test)
data_test_scaled[1:2] = scale_matrix(data_test_scaled[1:2], means, variances)

# KNN classification on test set
pred_test = knn(
  train = data_train_scaled[-3], test = data_test_scaled[-3], cl = data_train$Purchased, k = 5)

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
grid_x_0_1_scaled = scale_matrix(grid_x_0_1, means, variances)
grid_pred = knn(
  train = data_train_scaled[-3], test = grid_x_0_1_scaled, cl = data_train$Purchased, k = 5)

# Display plot
plot(
  data_test$Age, 
  data_test$EstimatedSalary,
  main = 'KNN',
  xlab = 'Age', 
  ylab = 'Sallary')
points(grid_x_0_1$Age, grid_x_0_1$EstimatedSalary, pch = 16, col = ifelse(grid_pred == 1, 'red', 'darkgreen'))
#points(data_test$Age, data_test$EstimatedSalary, pch = 16, col = ifelse(data_test$Purchased, 'red', 'darkgreen'))