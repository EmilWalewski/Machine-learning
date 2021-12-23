import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot
import pandas as pd


class Perceptron(object):

    def __init__(self, eta, n_iter, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.1, size=x.shape[1] + 1)
        self.errors = []
        for i in range(self.n_iter):
            error = 0
            for xi, target in zip(x, y):
                output = self.eta * (target - self.predict(xi))
                self.w[1:] += output * xi
                self.w[0] += output
                error += int(output != 0.0)
            self.errors.append(error)
        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)



    def plot_decision_regions(self, x, y):
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx1, yy1 = np.meshgrid(np.arange(x_min, x_max, 0.02),
                               np.arange(y_min, y_max, 0.02))
        z = self.predict(np.array([xx1.ravel(), yy1.ravel()]).T)
        z = z.reshape(xx1.shape)
        plot.contourf(xx1, yy1, z, alpha=0.3)

        for idx, cl in enumerate(np.unique(y)):
            plot.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8)


iris = load_iris()
x = iris.data[0:100, [0, 2]]
labels = iris.target[0:100]
y = np.where(labels == 0, -1, 1)


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
ppn.plot_decision_regions(x, y)

# plot.plot(range(1, 1 + len(ppn.errors)), ppn.errors, marker='o')
# plot.xlabel('Epoki')
# plot.ylabel('Liczba aktualizacji')
plot.show()
