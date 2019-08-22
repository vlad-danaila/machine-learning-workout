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
stds = apply(data_train[1:2], 2, sd)
feature_scaling = function(arrays, means, stds) {
  for (i in 1: length(arrays)) {
    arrays[i] = arrays[i] - means[i]
    arrays[i] = arrays[i] / stds[i]
  }
  arrays
}
data_train_scale = data.frame(data_train)
data_train_scale[1:2] = feature_scaling(data_train_scale[1:2], means, stds)
data_test_scale = data.frame(data_test)
data_test_scale[1:2] = feature_scaling(data_test_scale[1:2], means, stds)

# KNN classification on test set
pred_test = knn(
  train = data_train_scale[-3], test = data_test_scale[-3], cl = data_train_scale$Purchased, k = 5)

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
grid_x_0_1_scale = feature_scaling(grid_x_0_1, means, stds)
grid_pred = knn(
  train = data_train_scale[-3], test = grid_x_0_1_scale, cl = data_train_scale$Purchased, k = 15)

# Display plot
plot(
  data_test$Age, 
  data_test$EstimatedSalary,
  main = 'KNN',
  xlab = 'Age', 
  ylab = 'Sallary',
  xlim = range(grid_x_0),
  ylim = range(grid_x_1))
points(grid_x_0_1[, 1], grid_x_0_1[, 2], pch = 16, col = ifelse(grid_pred == 1, 'tomato', 'springgreen'))
points(data_test$Age, data_test$EstimatedSalary, pch = 16, col = ifelse(data_test$Purchased, 'darkred', 'darkgreen'))
