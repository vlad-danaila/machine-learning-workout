data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')
data = data[3:5]

# Make factors for y
purchased_levels = unique(data$Purchased)
data$Purchased = factor(data$Purchased, levels = purchased_levels)

# Scaling
data[1:2] = scale(data[1:2])

# Split train - test                        
library(caTools)
is_train = sample.split(data$Purchased, SplitRatio = 0.7)
data_train = subset(data, is_train)
data_test = subset(data, !is_train)

# Logistic regression
classifier = glm(formula = Purchased ~ ., data = data_train, family = binomial)
pred_prob = predict(classifier, data_test, type = 'response')
pred = ifelse(pred_prob > 0.5, 1, 0)

# Confusion matrix
cm = table(data_test$Purchased, pred == 1)
accuracy =  sum(diag(cm)) / sum(cm)
print(accuracy)

# Plot
#install.packages('ElemStatLearn')
library(ElemStatLearn)
grid_x_1 = seq(min(data_test[1] - 1), max(data_test[1] + 1), by = 0.03)
grid_x_2 = seq(min(data_test[2] - 1), max(data_test[2] + 1), by = 0.03)
grid = expand.grid(grid_x_1, grid_x_2)
colnames(grid) = colnames(data_test)[-3]
grid_pred = predict(classifier, grid, type = 'response')
grid_pred = ifelse(grid_pred > 0.5, 1, 0)
plot(data_test[, -3],
     main = 'Logistic Regression',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(grid_x_1), ylim = range(grid_x_2))
points(grid, pch = 5, col = ifelse(grid_pred == 1, 'springgreen3', 'tomato'))
points(data_test, pch = 21, bg = ifelse(data_test[, 3] == 1, 'green4', 'red3'))