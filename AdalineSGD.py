import numpy as np
import pandas as pd


class AdalineSGD(object):

    def __init__(self, eta, n_iter, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.is_initialized = False
        self.random_state = random_state

    def fit(self, x, y):
        self.init_weights(x.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._updateWights(x, target))
                avg_cost = cost.sum() / len(cost)
                self.cost_.append(avg_cost)
        return self

    def partial_fit(self, x, y):
        if not self.is_initialized:
            self._init_weights(x.shape[1])
        if x.ravel().shape[0] > 0:
            for xi, target in zip(x, y):
                self._updateWights(xi, target)
        else:
            self._updateWights(x, y)


    def _init_weights(self, x):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.1, size=x + 1)
        self.is_initialized = True

    def _shuffle(self, x, y):
        self.rgen = np.random.RandomState(self.random_state)
        r = self.rgen.permutation(len(y))
        print(f'Permutation value: {r}')
        return x[r], y[r]

    def _updateWights(self, x, y):
        output = self.activation(self.n_input(x))
        error = (y - output)
        self.w_[1:] += self.eta * np.dot(x, error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        self.cost_.append(cost)

    def activation(self, x):
        return x

    def n_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.activation(self.n_input(x)) >= 0.0, -1, 1)


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
x = data.iloc[:100, [0, 2]].values
target = data.iloc[:100, 4].values
y = np.where(target == 'Iris-setosa', 0, 1)
ada = AdalineSGD(eta=0.1, n_iter=10)
a, b = ada._shuffle(x, y)
print(a)
print(b)