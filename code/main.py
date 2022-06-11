import numpy as np
import pandas as pd
from numpy.random import shuffle

import nweb
from nweb import set_hyper_param as shp, NeuronWeb


# functions
def tts(a: list, b: list, k: float):
    c = list(zip(a, b))
    return c[:int(k * len(c))], c[int(k * len(c)):]


def new_nw(architecture: list[int]):
    res = nweb.NeuronWeb(architecture)
    print('nw was created')
    return res


# global variables
data_path = '../data/coffee.csv'

# data input
data = pd.read_csv(data_path)
low = np.array(data['Low'])
high = np.array(data['High'])
close = np.array(data['Close'])
opn = np.array(data['Open'])
vol = np.array(data['Volume'])

# data formatting
lm, hm, cm, om, vm = np.max(low) * 2, np.max(high) * 2, np.max(close) * 2, np.max(opn) * 2, np.max(vol) * 2
x = [np.array(
    list(low[i:i + 30] / lm) + list(high[i:i + 30] / hm) + list(close[i:i + 30] / cm) + list(
        opn[i:i + 30] / om) + list(vol[i:i + 30] / vm)) for i in range(data.shape[0] - 37)]
y = [np.array(
    list(low[i:i + 7] / lm) + list(high[i:i + 7] / hm) + list(close[i:i + 7] / cm) + list(opn[i:i + 7] / om) + list(
        vol[i:i + 7] / vm)) for i in range(30, data.shape[0] - 7)]
test, train = tts(x, y, 1 / 6)

# parameters
arch = [5 * 30, 100, 70, 50, 5 * 7]
shp(1, 10, 0)

# nw creation
nw = new_nw(arch)


# main part
def iterate(mdl: NeuronWeb):
    c = 0
    for tx, ty in test:
        c += nweb.cost_function(ty, mdl.iterate(tx))
    return c / len(test)


def fit(mdl: NeuronWeb, epochs: int):
    for _ in range(epochs):
        shuffle(train)
        for i in range(0, len(train), nweb.n):
            mdl.run(train[i:i + nweb.n], len(train))
