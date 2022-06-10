import pandas as pd
import numpy as np
import nweb

data_path = '../data/coffee.csv'

data = pd.read_csv(data_path)
low = np.array(data['Low'])
high = np.array(data['High'])
close = np.array(data['Close'])
opn = np.array(data['Open'])
vol = np.array(data['Volume'])

nweb.set_hyper_param(1, 10, 0)
