import cupy as np
import numpy.random
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
data_path = '../data/coffee.csv'
saved_nw_path = '../saves'
input_days = 60
output_days = 1
amount_of_test_data = 1 / 6
costs_to_show = []

# data input
data = pd.read_csv(data_path)
low = np.array(data['Low'], dtype=float)
high = np.array(data['High'], dtype=float)
close = np.array(data['Close'], dtype=float)
opn = np.array(data['Open'], dtype=float)
vol = np.array(data['Volume'], dtype=float)

# data formatting
f = 200
x = [np.array(
    list((low[i:i + input_days] - opn[i]) / f) + list((high[i:i + input_days] - opn[i]) / f) +
    list((close[i:i + input_days] - opn[i]) / f) + list((opn[i:i + input_days] - opn[i]) / f) +
    list(np.power(vol[i:i + input_days], 1 / 2) / f))
    for i in range(data.shape[0] - input_days - output_days)]
y = [np.array(
    list((low[i:i + output_days] - opn[i - input_days]) / f) + list((high[i:i + output_days] - opn[i - input_days]) / f)
    + list((close[i:i + output_days] - opn[i - input_days]) / f) +
    list((opn[i:i + output_days] - opn[i - input_days]) / f) + list(np.power(vol[i:i + output_days], 1 / 2) / f))
    for i in range(input_days, data.shape[0] - output_days)]
np.cuda.Stream.null.synchronize()
test, train = tts(x, y, amount_of_test_data)
x = y = None

# parameters
arch = [5 * input_days, 5000, 3000, 1000, 5 * output_days]
shp(1e-5, 1, 0)

# nw creation
nw = new_nw(arch)


# main part
def iterate(mdl: NeuronWeb, dt=None):
    if dt is None:
        dt = test
    c = 0
    for tx, ty in dt:
        c += float(np.sum(nweb.cost_function(ty, mdl.iterate(tx)[0]))) / len(ty)
    np.cuda.Stream.null.synchronize()
    return c / len(dt)


def fit(mdl: NeuronWeb, epochs: int):
    for _ in range(epochs):
        shuffle(train)
        for i in range(0, len(train), nweb.n):
            mdl.run(train[i:i + nweb.n], len(train))
        np.cuda.Stream.null.synchronize()
        global costs_to_show
        costs_to_show += [iterate(mdl)]
        print(costs_to_show[-1])


def test_res(a: int):
    return np.abs((test[a][1] - nw.iterate(test[a][0])[0]) / test[a][1])


def show(yh, xl=0, xh=len(costs_to_show), yl=0):
    plt.ylim(yl, yh)
    plt.xlim(xl, xh)
    plt.plot(costs_to_show)
    plt.show()


costs_to_show.append(iterate(nw))
