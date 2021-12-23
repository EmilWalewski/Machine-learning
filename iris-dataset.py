import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1] + 1)
        self.errors = []
        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def predict(self, x):
        return np.where(self.network_act(x) >= 0, 1, -1)

    def network_act(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def plot_decision_regions(self, x, y, resolution=0.02):
        x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        print(np.arange(x1_min, x1_max, resolution).shape)
        print(np.arange(x2_min, x2_max, resolution).shape)
        print(xx1.ravel().shape)
        print(xx2.ravel().shape)
        z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z = z.reshape(xx1.shape)
        plot.contourf(xx1, xx2, z, alpha=0.3)

        for idx, cl in enumerate(np.unique(y)):
            plot.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8)


data_frame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
labels = data_frame.iloc[0:100, 4].values
y = np.where(labels == 'Iris-setosa', -1, 1)
x = data_frame.iloc[0:100, [0, 2]].values

# sepal and petal features pots
# plot.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
# plot.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')

perceptron = Perceptron()
perceptron.fit(x, y)
# error plots
# plot.plot(np.arange(1, len(perceptron.errors)+1), perceptron.errors)
# perceptron.plot_decision_regions(x, y)
# plot.show()

arr1 = np.array([1, 2, 3])
arr2 = np.array(['J', 'A', 'B', 'C', 'A', 'D', 'B'])

zipped = zip(arr1, arr2)

# print(tuple(zipped))/

# print(tuple(enumerate(arr2)))

to_mesh = np.arange(0, 5, 0.5)
to_mesh2 = np.arange(0, 4, 0.5)
arr1x1, arr2x1 = np.meshgrid(to_mesh, to_mesh2)
print(arr1x1)
print(arr2x1)
# plot.plot(arr1x1, arr2x1)
# plot.show()
frame = np.array([arr1x1.ravel(), arr2x1.ravel()]).T
print(arr1x1.shape)
print(arr2x1.shape)
print(frame.shape)

target = np.array(np.random.randint(0, 2, 100))
x = np.array([np.random.randint(0, 10, 100), np.random.randint(0, 10, 100)]).reshape(100, 2)
print(target)
t = []
for idx, cl in enumerate(np.unique(target)):
    t.append(x[target == cl, 0])
    plot.scatter(x=x[target == cl, 0], y=x[target == cl, 1], alpha=0.8)

# plot.show()
