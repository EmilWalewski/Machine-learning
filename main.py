import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plot

class Perceptron(object):

    def __init__(self, eta=0.01, n_inter=50, random_state=1):
        self.eta = eta
        self.n_inter = n_inter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_inter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


ppn = Perceptron(eta=0.1, n_inter=10)
# source = np.array([[1, 2, 3], [4, 5, 6], [0, 8, 9]])
# target = np.array([1, 1, -1, 1, 1, -1])
iris = load_iris()
x = iris.data[0:100, [0, 2]]
y = iris.target[0:100]
ppn.fit(x, y)
plot.plot(range(1, 1 + len(ppn.errors_)), ppn.errors_, marker='o')
plot.xlabel('Epoki')
plot.ylabel('Liczba aktualizacji')
plot.show()

