''''

Forward Multi-Assets

'''
import numpy as np
import pandas as pd
from itertools import product
import scipy.optimize as opt
import myprint
import ds

def get_util(alpha_vector, x, model_vector, price_vector):
    '''
    @param alpha_vector: np.array. vector for alpha [a0_1, a0_2, ..]
    @param model_vector: np.array. vector for model [(u1-1, d1-1, p1, 1-p1), (u2, d2, p2, 1-p2), ..]
    @param price_vector: np.array. vector for Price [S_1, S_2, ..]
    @return: 0 - Utility Value (because we want to use fmin in optimization)
    '''
    ret = 0
    n = len(alpha_vector)
    for w in list(product(range(2), repeat=n)):
        S_diff = np.array([model_vector[i][w[i]] for i in range(n)]) * price_vector
        X = x + sum(alpha_vector * S_diff)
        p = sum(np.array([model_vector[i][2 + w[i]] for i in range(n)]))
        ret += np.exp(-X) * p
    return ret


'''
Read Data
'''
col_names = ['AAPL', 'AMZN', 'GOOG', 'NFLX', 'TSLA', 'WMT']
weekly_Closes = pd.DataFrame(columns = col_names)
for name in col_names:
    weekly_Closes[name] = ds.get_CloseFromRaw(name)


n = len(weekly_Closes)
ini_wealth = 0
x = ini_wealth
x_list = [x]
alpha_list = []
diff_list = []
for i in range(90, 100):
    train = weekly_Closes.iloc[0:i+1]
    model_vector = [ds.get_model(np.array(train.iloc[:,j])) for j in range(len(col_names))]
    prices_vector = np.array(train.iloc[i])
    result = opt.fmin(get_util, np.zeros(len(col_names)), args=(x, model_vector, prices_vector))
    alphas = result
    alpha_list.append(alphas) # record investment strategy
    real_Next_price = np.array(weekly_Closes.iloc[i+1])
    diff = sum(np.array(alphas) * (real_Next_price - prices_vector))
    x += diff
    diff_list.append(diff)
    x_list.append(x)
    # print(result)
    # print("complete iteration, current wealth is: " + str(x))

print(x_list[-1])
print(alpha_list)



import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.plot(diff_list)
plt.show()