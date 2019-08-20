data = read.csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
data = data[-1]

# Feature scalling
data = scale(data)

# SVR
install.packages('e1071')