dataset = read.csv("C:\\DOC\\Workspace\\Machine Learning A-Z Template Folder\\Part 1 - Data Preprocessing\\Data.csv")

# Missing data
fill_missing_data = function(features) {
  mean = mean(features, na.rm = TRUE)
  ifelse(is.na(features), mean, features)
}
dataset$Age = fill_missing_data(dataset$Age)
dataset$Salary = fill_missing_data(dataset$Salary)

# Categorical data
make_factors = function(features) {
  levels_features = levels(features)
  factor(features, levels = levels_features, labels = seq(length(levels_features)))
}
dataset$Country = make_factors(dataset$Country)
dataset$Purchased = factor(dataset$Purchased, levels = c('No', 'Yes'), labels = c(0, 1))

# Feature scaling
dataset[, 2:3] = scale(dataset[, 2:3])

# Split
#install.packages('caTools')
library('caTools')
dataset_split = sample.split(dataset$Purchased, SplitRatio = 0.7)
train = subset(dataset, dataset_split)
test = subset(dataset, !dataset_split)
