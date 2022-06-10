import numpy as np
from numpy import ndarray

# скорость обучения
e = 0.25
# размер пакета
n = 10
# коэффициент регуляризации
lmbda = 0.1


def set_hyper_param(ee: float, nn: int, l: float):
    e = ee
    n = nn
    lmbda = l


class NeuronWeb:

    def __init__(self, a: list):
        self.a = a
        self.weights = [np.random.randn(a[i + 1], a[i]) / np.sqrt(a[i]) for i in range(len(a) - 1)]
        self.biases = [np.random.randn(i) for i in a[1:]]

    def run(self, mini_batch: ndarray, size: int):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        for data, i in mini_batch:
            ndw, ndb = self.correct(i, self.iterate(data)[1])
            for j in range(len(db)):
                dw[j] += ndw[j]
            for j in range(len(db)):
                db[j] += ndb[j]
        for i in range(len(dw)):
            self.weights[i] = (1 - lmbda * e / size) * self.weights[i] - (e / n) * dw[i]
        for i in range(len(db)):
            self.biases[i] -= (e / n) * db[i]

    def iterate(self, data: ndarray):
        acts = [np.zeros(i) for i in self.a]
        acts[0] = data
        for l in range(len(self.biases)):
            data = sig(np.dot(self.weights[l], np.transpose(data)) + self.biases[l])
            acts[l + 1] = data
        return np.argmax(data), acts

    def correct(self, i: int, acts: list[ndarray]):
        i = np.array([(1 if i == j else 0) for j in range(10)])
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        i = cost_derivative(acts[-1], i)
        for l in range(1, len(self.biases) + 1):
            db[-l] = i
            dw[-l] = np.dot(np.reshape(i, (len(i), 1)), np.reshape(acts[-l - 1], (1, len(acts[-l - 1]))))
            i = np.dot(np.transpose(self.weights[-l]), i) * sig_derivative(acts[-l - 1])
        return dw, db


#   dC
#   --
#   dz
#
# binary entrophy
# c=-(y*ln(sig(z))+(1-y)ln(1-sig(z)))
def cost_derivative(a: ndarray, y: ndarray):
    return a - y


def sig(z: ndarray):
    return 1 / (1 + np.exp(-z))


def sig_derivative(a: ndarray):
    return a * (1 - a)
