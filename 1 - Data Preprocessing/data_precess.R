dataset = read.csv("C:\\DOC\\Workspace\\Machine Learning A-Z Template Folder\\Part 1 - Data Preprocessing\\Data.csv")

# Missing data
mean_age = mean(dataset$Age, na.rm = TRUE)
mean_salary = mean(dataset$Salary, na.rm = TRUE)

is_na_age = is.na(dataset$Age)
is_na_salary = is.na(dataset$Salary)

dataset$Age = ifelse(is_na_age, mean_age, dataset$Age)
dataset$Salary = ifelse(is_na_salary, mean_salary, dataset$Salary)