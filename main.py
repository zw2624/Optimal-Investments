import pandas as pd
import numpy as np
import time
from myprint import print_Runtime
import ds
import preference
import pandas as pd
from itertools import product
import scipy.optimize as opt


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
weekly_Closes  = pd.DataFrame(columns = col_names)
for name in col_names:
    weekly_Closes[name] = ds.get_CloseFromRaw(name)


n = len(weekly_Closes)
ini_wealth = 0
x = ini_wealth
x_list = [x]
alpha_list = []
for i in range(90, 100):
    train = weekly_Closes.iloc[0:i+1]
    model_vector = [ds.get_model(np.array(train.iloc[:,j])) for j in range(len(col_names))]
    prices_vector = np.array(train.iloc[i])
    #print("Start optimization")
    result = opt.fmin(get_util, np.zeros(len(col_names)), args=(x, model_vector, prices_vector))
    #print("Found result")
    alphas = result[0]
    util = 0 - result[1]
    alpha_list.append(alphas) # record investment strategy
    real_Next_price = np.array(weekly_Closes.iloc[i+1])
    x += sum(np.array(alphas) * (real_Next_price - prices_vector))
    x_list.append(x)
    print("complete iteration, current wealth is: " + str(x))

print(x_list[-1])



# all_stock = pd.read_excel("./data/raw.xlsx")
# dates = np.array(all_stock['Date'])
# dates = dates[::-1]
#
#
# prices = np.array(all_stock['Adj Close'].values)
# prices = prices[::-1] # reverse the list; make the price f
# all_Close_prices = prices[2::5]  # length = 50
#
#
#
#
# '''Weekly Investment'''
# '''
# train/test split.
# Totally there are 50 periods. We use first 40 as training and last 10 as validation
# '''
# train_prices_weekly = prices[0:3 + 39 * 5 - 1]
# test_prices_weekly = prices[2 + 39 * 5 - 1:]
# period = 10
#
# '''
# simulation
# '''
# weekly_Backward_alpha = np.array([])
# weekly_Forward_alpha = np.array([])
# weekly_Backward_h = np.array([])
# weekly_Forward_h = np.array([])
# earning_b = np.array([])
# earning_f = np.array([])
# gamma = 1
# time_back = []
# time_forward = []
#
#
#
# for i in range(period):
#     info = ds.get_parameters(train_prices_weekly, 2, 5)  # calculate market estimates
#     # disounted_train = parameter.get_discounted(train_prices_weekly, 1+info[2])  # discount the price
#     weekly_market_model = ds.get_market_model(info[1], 5, True, train_prices_weekly) # get market model
#     S0 = train_prices_weekly[-1] # get current stock price as S0
#
#     t1 = time.clock()
#     p_b = preference.solve_Backward(10 - i, weekly_market_model, S0, gamma) # calculate Backward Preference
#     t2 = time.clock()
#     p_f = preference.solve_Forward(weekly_market_model, S0, gamma)# calculate Forward Preference
#     t3 = time.clock()
#     time_back.append(t2 - t1)
#     time_forward.append(t3 - t2)
#
#
#     weekly_Backward_alpha = np.append(weekly_Backward_alpha, p_b[1]) # mark investment - backward
#     weekly_Forward_alpha = np.append(weekly_Forward_alpha, p_f[1]) # mark investment - forward
#     weekly_Backward_h = np.append(weekly_Backward_h, p_b[0]) # mark investment - backward
#     weekly_Forward_h = np.append(weekly_Forward_h, p_f[0]) # mark investment - forward
#
#     real_C = all_Close_prices[i + 40]  # Get new Close Price
#
#     earning_b = np.append(earning_b, weekly_Backward_alpha * (real_C - S0)) # record investment result
#     earning_f = np.append(earning_f, weekly_Forward_alpha * (real_C - S0)) # record investment result
#
#     train_prices_weekly = np.append(train_prices_weekly, test_prices_weekly[0:4]) # Update Training set
#     test_prices_weekly = test_prices_weekly[5:] # Update Testing set
#
# weekly_Backward_h
# weekly_Forward_h
# #
# print(time_back)
# print(time_forward)
# print_Runtime(time_back, time_forward)
#
# # import myprint
# # myprint.print_Utility(True, weekly_Backward_h, gamma, 0, 1, 0.001)
# # myprint.print_Utility(False, weekly_Forward_h, gamma, 0, 1, 0.001)
#
#
#
#
# '''
# plt.plot(range(len(all_Close_prices)), all_Close_prices)
# plt.show()
#
# plt.plot(range(len(prices)), prices)
# plt.show()
# '''
#
# '''
# visualization
# '''
#
#
#
# '''
# debug
# '''
#
#
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.set_ylabel('Stock Price')
ax2.set_ylabel('Backward Alpha')
ax3.set_ylabel('Forward Alpha')

y = all_Close_prices
x = [i for i in range(40)]
y_b = [0]*40 + list(weekly_Backward_alpha)
y_f = [0]*40 + list(weekly_Forward_alpha)
stock, = ax1.plot(x, [min(y), max(y)]*20)
ax2.set_ylim(0, 1e-3)
ax3.set_ylim(0, 1e-3)
back, = ax2.plot(x, [min(y_b), max(y_b)]*20)
forward, = ax3.plot(x, [min(y_f), max(y_f)]*20)

def init():
    stock.set_ydata([0] * len(x))
    back.set_ydata([0] * len(x))
    forward.set_ydata([0] * len(x))
    return stock,back, forward

def animate(i):
    stock.set_ydata(y[i:i+40]) # update the data.
    back.set_ydata(y_b[i:i+40])  # update the data.
    forward.set_ydata(y_f[i:i+40])  # update the data.
    return stock, back, forward


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=300, blit=True, save_count=10)

ani.save("movie.mp4")
# plt.show()


