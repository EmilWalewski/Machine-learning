import numpy as np
from scipy import sparse
from sklearn import datasets

random_array = np.array([np.random.randint(0, 10, 3), np.random.randint(0, 10, 3), np.random.randint(0, 10, 3),
                         np.random.randint(0, 10, 3)])
# normal_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
print(random_array)
random_y_array = np.array([2, 3, 4, 6])
r1 = random_array[random_y_array < 7, 1]
# r1 = np.where(random_array > 5, 1, -1)
print(r1)

matrix = np.array([[0, 0, 1, 1, 0, 0, 9, 0, 1, 0],
                   [1, 0, 1, 0, 1, 7, 0, 1, 1, 1],
                   [1, 1, 5, 1, 0, 1, 1, 0, 1, 0]])
matrix_spares = sparse.csc_matrix(matrix)
# print(matrix_spares)

print(np.max(matrix, axis=0))

np.random.seed(0)

array_with_seed_1 = np.random.rand(5)

print(array_with_seed_1)

array_random_int = np.random.randint(4, 8, 5)

print(array_random_int)

array_random_uniform = np.random.uniform(1.0, 2.0, 5)

print(array_random_uniform)

array_normal = np.random.normal(1.0, 1.0, 5)

print(array_normal)


iris = datasets.load_iris()

print(iris.data[0])
print(iris.target)


