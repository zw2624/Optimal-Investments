''''

Forward Multi-Assets

'''
import numpy as np
import pandas as pd
from itertools import product
import scipy.optimize as opt
import myprint
import ds
from bokeh.plotting import output_file, figure, show, save

all_stock = pd.read_csv("./data/GSPC.csv")
prices = np.array(all_stock['Close'].values)
all_Close_prices = prices[2::5]

def get_util(alpha, x, model_vector, price):
    ret = 0
    V = model_vector[0]
    p1 = model_vector[1]
    p2 = model_vector[2]
    p3 = model_vector[3]
    S_diff = np.array([V, 0, -V]) * price
    X = x + alpha * S_diff
    ret += np.exp(-X) * np.array([p1, p2, p3])
    return ret

period = 12
total = len(all_Close_prices)
train_prices_weekly = all_Close_prices[0:(total-period)]
test_prices_weekly = all_Close_prices[(total-period):]


ini_wealth = 0
x = ini_wealth
x_list = [x]
alpha_list = []
earning_tri = np.array([])
for i in range(period-1):
    info = ds.get_parameters(train_prices_weekly, 5)
    weekly_market_model = ds.get_tri_market_model(info[1], 5, 1.2)
    S0 = train_prices_weekly[-1]
    result = opt.fmin(get_util, 0, args=(x, weekly_market_model, S0))
    alpha_list.append(result[0]) # record investment strategy
    real_C = all_Close_prices[i + (total - period)]
    x += result[0] * (real_C - S0)
    earning_tri = np.append(earning_tri, result[0] * (real_C - S0))

