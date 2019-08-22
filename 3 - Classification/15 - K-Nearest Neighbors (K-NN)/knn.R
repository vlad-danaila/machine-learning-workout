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