import numpy as np
import pandas as pd
from numpy.random import shuffle
import matplotlib.pyplot as plt
import nweb
from nweb import set_hyper_param as shp
from nweb import NeuronWeb


# functions
def tts(a: list, b: list, k: float):
    c = list(zip(a, b))
    shuffle(c)
    return c[:int(k * len(c))], c[int(k * len(c)):]


def new_nw(architecture: list[int]):
    res = nweb.NeuronWeb(architecture)
    print('nw was created')
    return res


def random_cost():
    c = 0
    for tx, ty in test:
        c += np.sum(nweb.cost_function(ty, 0.99 * np.random.rand(len(ty)) + 0.001)) / len(ty)
    return c / len(test)


# global variables
data_path = '../data/smoke.csv'
saved_nw_path = '../saves'
test_costs_to_show = []
train_costs_to_show = []
dtype = np.float64

# data input
data = pd.read_csv(data_path)

# data formatting
for i in data.columns:
    if i == 'Fire Alarm': continue
    data[i] -= 3 / 2 * np.min(data[i]) - 1
    data[i] /= 3 / 2 * np.max(data[i]) + 1
y = data['Fire Alarm'].tolist()
z = data.drop('Fire Alarm', axis=1)
for i in range(len(y)):
    y[i] = np.array(y[i], dtype=dtype).reshape(1)
x = list(map(lambda a: np.array(a[1], dtype=dtype), z.iterrows()))
z = None
del z
test, train = tts(x, y, 1 / 6)
x = y = None
del x, y

# parameters
arch = [data.columns.size - 1, 1000, 100, 1]
shp(1e-4, 1, 0)

# nw creation
nw = new_nw(arch)


# main part
def iterate(mdl: NeuronWeb, dt=None):
    if dt is None:
        dt = test
    c = 0
    for tx, ty in dt:
        c += float(np.sum(nweb.cost_function(ty, mdl.iterate(tx)[0])))
    # np.cuda.Stream.null.synchronize()
    return c / len(dt)


def fit(mdl: NeuronWeb, epochs: int):
    for _ in range(epochs):
        shuffle(train)
        for i in range(0, len(train), nweb.n):
            mdl.run(train[i:i + nweb.n], len(train))
        # np.cuda.Stream.null.synchronize()
        global test_costs_to_show, train_costs_to_show
        test_costs_to_show += [iterate(mdl, test)]
        train_costs_to_show += [0]
        print(test_costs_to_show[-1])


def count(mdl: NeuronWeb, dt=None):
    if dt is None:
        dt = test
    c = 0
    for tx, ty in dt:
        c += (1 if (int(round(float(mdl.iterate(tx)[0][0]))) == int(ty)) else 0)
    return c


def show(yh, xl=0, xh=None, yl=0):
    if xh is None:
        xh = len(test_costs_to_show)
    plt.ylim(yl, yh)
    plt.xlim(xl, xh)
    plt.plot(test_costs_to_show, color='#002dbf')
    plt.plot(train_costs_to_show, color='#ff8800')
    plt.show()


test_costs_to_show.append(iterate(nw, test))
train_costs_to_show.append(iterate(nw, train))
