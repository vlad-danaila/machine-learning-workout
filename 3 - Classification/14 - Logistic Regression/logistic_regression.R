data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')
data = data[3:5]

# Make factors for y
purchased_levels = unique(data$Purchased)
data$Purchased = factor(data$Purchased, levels = purchased_levels, labels = seq(length(purchased_levels)))

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

