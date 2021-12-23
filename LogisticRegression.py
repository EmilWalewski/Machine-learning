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
        def fit(self, x, y):
            rgen = np.random.RandomState(self.random_state)
            self.w_ = rgen.normal(loc=0.0, scale=0.1, size=x.shape[1] + 1)
            self.cost_ = []
            for i in range(self.n_iter):
                output = self.activation(self.n_input(x))
                error = y - output
                self.w[1:] = self.eta * x.T.dot(error)
                self.w_[0] = self.eta * error
                cost = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output)))
                self.cost_.append(cost)
            return self

        return self

    def activation(self, x):
        return 1 / (1 - np.exp(-np.clip(x, -250, 250)))

    def n_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.activation(self.n_input(x)) >= 0.0, 1, 0)


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
x = data.iloc[:100, [0, 2]].values
target = data.iloc[:100, 4].values
y = np.where(target == 'Iris-setosa', 0, 1)
logistinc_regression = AdalineSGD(eta=0.1, n_iter=10)
