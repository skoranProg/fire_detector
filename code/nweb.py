import cupy as np
from cupy import ndarray
import os

# скорость обучения
e = 0.25
# размер пакета
n = 10
# коэффициент регуляризации
lmb = 0.1


def set_hyper_param(ee: float, nn: int, l: float):
    global e
    e = ee
    global n
    n = nn
    global lmb
    lmb = l


class NeuronWeb:

    def __init__(self, a: list):
        self.a = a
        self.weights = [np.random.randn(a[i + 1], a[i]) / np.sqrt(a[i]) for i in range(len(a) - 1)]
        self.biases = [np.random.randn(i) for i in a[1:]]

    def save(self, path, nw_name):
        try:
            os.mkdir(f'{path}/{nw_name}')
            np.save(f'{path}/{nw_name}/arch.npy', np.array(self.a))
            np.save(f'{path}/{nw_name}/params.npy', np.array([e, float(n), lmb]))
            for i in range(len(self.weights)):
                np.save(f'{path}/{nw_name}/weights_{i}.npy', self.weights[i])
            for i in range(len(self.biases)):
                np.save(f'{path}/{nw_name}/biases_{i}.npy', self.biases[i])
        except FileExistsError:
            inp = input('Do you really wanna rewrite old nw?(y/n)').lower()
            if inp == 'y':
                np.save(f'{path}/{nw_name}/arch.npy', np.array(self.a))
                np.save(f'{path}/{nw_name}/params.npy', np.array([e, float(n), lmb]))
                for i in range(len(self.weights)):
                    np.save(f'{path}/{nw_name}/weights_{i}.npy', self.weights[i])
                for i in range(len(self.biases)):
                    np.save(f'{path}/{nw_name}/biases_{i}.npy', self.biases[i])
            elif inp != 'n':
                print('Can you just choose valid option?!')
        except FileNotFoundError:
            print('Error!!!\nWrong path!!!')

    @staticmethod
    def load(path, nw_name):
        try:
            nw = NeuronWeb([])
            nw.a = np.load(f'{path}/{nw_name}/arch.npy').tolist()
            params = np.load(f'{path}/{nw_name}/params.npy')
            set_hyper_param(float(params[0]), int(params[1]), float(params[2]))
            nw.weights = []
            nw.biases = []
            for i in range(len(nw.a) - 1):
                nw.weights.append(np.load(f'{path}/{nw_name}/weights_{i}.npy'))
                nw.biases.append(np.load(f'{path}/{nw_name}/biases_{i}.npy'))
            return nw
        except FileNotFoundError:
            print('Error!!!\nWrong path!!!')

    def run(self, mini_batch: list[tuple[ndarray, ndarray]], size: int):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        for data, i in mini_batch:
            ndw, ndb = self.correct(i, self.iterate(data)[1])
            for j in range(len(db)):
                dw[j] += ndw[j]
            for j in range(len(db)):
                db[j] += ndb[j]
        for i in range(len(dw)):
            self.weights[i] = (1 - lmb * e / size) * self.weights[i] - (e / n) * dw[i]
        for i in range(len(db)):
            self.biases[i] -= (e / n) * db[i]
        np.cuda.Stream.null.synchronize()

    def iterate(self, data: ndarray):
        acts = [np.zeros(i) for i in self.a]
        acts[0] = data
        for l in range(len(self.biases)):
            data = tanh(np.dot(self.weights[l], np.transpose(data)) + self.biases[l])
            acts[l + 1] = data
        return data, acts

    def correct(self, i: ndarray, acts: list[ndarray]):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        i = cost_derivative(acts[-1], i)
        for l in range(1, len(self.biases) + 1):
            np.cuda.Stream.null.synchronize()
            db[-l] = i
            dw[-l] = np.dot(np.reshape(i, (len(i), 1)), np.reshape(acts[-l - 1], (1, len(acts[-l - 1]))))
            i = np.dot(np.transpose(self.weights[-l]), i) * tanh_derivative(acts[-l - 1])
        return dw, db


#   dC
#   --
#   dz
#
# binary entrophy
# c=-(y*ln(sig(z))+(1-y)ln(1-sig(z)))
#
# square function actually
def cost_function(y: ndarray, a: ndarray):
    return (a - y) ** 2


def cost_derivative(a: ndarray, y: ndarray):
    return 2 * (a - y) * tanh_derivative(a)


# tanh actually
def tanh(z: ndarray):
    return np.tanh(z)


def tanh_derivative(a: ndarray):
    return 1 - a * a
