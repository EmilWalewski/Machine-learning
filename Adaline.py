import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


class Adaline(object):

    def __init__(self, eta, n_iter, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        self.w_ = self.random(x)
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.activation(self.n_input(x))
            error = (y - output)
            self.w_[1:] += self.eta * np.dot(x.T, error)
            self.w_[0] += self.eta * error.sum()
            cost = (error ** 2).sum() / 2.00
            self.cost_.append(cost)
        return self

    def random(self, x):
        rand = np.random.RandomState(self.random_state)
        return rand.normal(loc=0.0, scale=0.1, size=x.shape[1] + 1)

    def activation(self, x):
        return x

    def n_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.activation(self.n_input(x)) >= 0.0, -1, 1)


def plot_decision_regions(x, y, classifier, resolution=0.2):
    markers = ('x', 's', 'o', '^')
    colors = ('gray', 'red', 'blue', 'green')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
    z = z.reshape(x1.shape)
    plt.contourf(x1, x2, z, cmap=cmap)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, marker=markers[idx],
                    label=cl, edgecolor='black')


frame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
x = frame.iloc[:100, [0, 2]].values
labels = frame.iloc[:100, 4].values
y = np.where(labels == 'Iris-setosa', -1, 1)

x_std = np.copy(x);
x_std[:, 0] = (x_std[:, 0] - x_std[:, 0].mean()) / x_std[:, 0].std()
x_std[:, 1] = (x_std[:, 1] - x_std[:, 1].mean()) / x_std[:, 1].std()

ada = Adaline(eta=0.1, n_iter=15)
ada.fit(x, y)
z = x[:3]
yz = y[:3]
b = z[z <= 5.1]
print(z)
print(yz)
print(b)
plot_decision_regions(x, y, classifier=ada)


# plt.show()