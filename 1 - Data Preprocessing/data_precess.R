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
dataset$Purchased = make_factors(dataset$Purchased)

