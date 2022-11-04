#https://www.udemy.com/course/learn-machine-learning-in-21-days/learn/lecture/23665398#overview
import numpy as np

row_vector = np.array([10,15,25])

column_vector = np.array([[10,15,25]])

print(row_vector)

matrix = np.array([[10,20,30],[1,2,3],[51,23,53]])
print(matrix)
print(matrix[2,1])
print(matrix[:2,:])
print("row and columns", matrix.shape)
print("total elemens", matrix.size)
print("Dimension",matrix.ndim)
print(np.max(matrix))
print(np.min(matrix))
print(np.max(matrix,axis = 0))
print(np.min(matrix,axis = 1))
print("Promedio:", np.mean(matrix))
print(matrix.reshape(9,1))
print(matrix.reshape(1,9))
print(matrix.reshape(3,3))
print("Darlo vuelta\n ", matrix.T)  