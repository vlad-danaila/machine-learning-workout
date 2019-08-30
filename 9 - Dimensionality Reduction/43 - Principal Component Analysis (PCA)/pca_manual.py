# -*- coding: utf-8 -*-

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# define a matrix, columns are data elements, rows are features
# 3 elements by 2 features
M = array([[1, 2], [3, 4], [5, 6]]).T
'''
array([[1, 3, 5],
       [2, 4, 6]])
'''

# calculate the mean of each column
mean = mean(M, axis = 1).reshape(2, 1)
'''
array([[3.],
       [4.]])
'''

# center the matrix
M_centered = M - mean
'''
array([[-2.,  0.,  2.],
       [-2.,  0.,  2.]])
'''

# covariance matrix
C = cov(M_centered)
'''
array([[4., 4.],
       [4., 4.]])
'''

# eigen values & vectors
values, vectors = eig(C)
'''
values
array([8., 0.])

vectors
array([[ 0.70710678, -0.70710678],
       [ 0.70710678,  0.70710678]])

vectors.T
array([[ 0.70710678,  0.70710678],
       [-0.70710678,  0.70710678]])
'''

# transform data
P = vectors.T.dot(M_centered)
''' 
array([[-2.82842712,  0.        ,  2.82842712],
       [ 0.        ,  0.        ,  0.        ]])
each coulmn is an element, the rows are features
simply transpose to get a classic representation
'''