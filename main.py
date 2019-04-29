import numpy as np
import pandas as pd
from itertools import product
import scipy.optimize as opt
import myprint
import ds
''''

Backward and Forward Comparison

'''


import time
from myprint import print_Runtime
import ds
import preference

all_stock = pd.read_csv("./data/GSPC.csv")
prices = np.array(all_stock['Close'].values)
all_Close_prices = prices[2::5]


'''Weekly Investment'''
'''
train/test split.
Totally there are 50 periods. We use first 40 as training and last 10 as validation
'''
period = 12
total = len(all_Close_prices)
train_prices_weekly = all_Close_prices[0:(total-period)]
test_prices_weekly = all_Close_prices[(total-period):]


'''
simulation
'''
weekly_Backward_alpha = np.array([])
weekly_Forward_alpha = np.array([])
weekly_Backward_h = np.array([])
weekly_Forward_h = np.array([])
earning = np.array([])
earning_b = np.array([])
earning_f = np.array([])
gamma = 1
time_back = []
time_forward = []
last_h = 0

'''

'''
info = ds.get_parameters(train_prices_weekly, 5)
market_model = ds.get_market_model(info[1], 5, True, train_prices_weekly)

time_origin = []
t1 = time.clock()
backward_matrix = preference.get_backward_matrix(period, market_model)
t2 = time.clock()
const_time = t2-t1

S = train_prices_weekly[-1]
S_assume = train_prices_weekly[-1]
u = market_model[0] + 1
d = market_model[1] + 1
pu = market_model[2]
pd = market_model[3]
q = (1 - d) / (u - d)
index = 0
for i in range(period-1):
    t1 = time.clock()
    h_u = backward_matrix[i + 1][index + 1]
    h_d = backward_matrix[i + 1][index + 1]
    alpha = (np.log(pu / q) - np.log(pd / q) - (h_u - h_d)) / (gamma * S_assume * (u - d))
    next_S = test_prices_weekly[i]
    t2 = time.clock()
    time_origin.append(t2 - t1)
    earning = np.append(earning, alpha * (next_S - S))
    index = index if next_S > S else index + 1
    S_assume = S_assume * u if next_S > S else S_assume * d
    S = next_S
time_origin[0] += const_time

'''

'''
for i in range(period-1):
    info = ds.get_parameters(train_prices_weekly, 5)  # calculate market estimates
    # disounted_train = parameter.get_discounted(train_prices_weekly, 1+info[2])  # discount the price
    weekly_market_model = ds.get_market_model(info[1], 5, True, train_prices_weekly) # get market model
    S0 = train_prices_weekly[-1] # get current stock price as S0

    t1 = time.clock()
    p_b = preference.solve_Backward(10, weekly_market_model, S0, gamma) # calculate Backward Preference
    t2 = time.clock()
    p_f = preference.solve_Forward(weekly_market_model, S0, gamma)# calculate Forward Preference
    t3 = time.clock()
    time_back.append(t2 - t1)
    time_forward.append(t3 - t2)


    weekly_Backward_alpha = np.append(weekly_Backward_alpha, p_b[1]) # mark investment - backward
    weekly_Forward_alpha = np.append(weekly_Forward_alpha, p_f[1]) # mark investment - forward
    weekly_Backward_h = np.append(weekly_Backward_h, p_b[0]) # mark investment - backward
    weekly_Forward_h = np.append(weekly_Forward_h, p_f[0]) # mark investment - forward

    real_C = all_Close_prices[i + (total-period)]  # Get new Close Price
    earning_b = np.append(earning_b, weekly_Backward_alpha[-1] * (real_C - S0)) # record investment result
    earning_f = np.append(earning_f, weekly_Forward_alpha[-1] * (real_C - S0)) # record investment result

    train_prices_weekly = np.append(train_prices_weekly, test_prices_weekly[0]) # Update Training set
    test_prices_weekly = test_prices_weekly[1:] # Update Testing set

weekly_Forward_h_cum = [sum(weekly_Forward_h[0:i]) for i in range(len(weekly_Forward_h))]


t_list_1 = [[time_back, "Backward with update"], [time_origin, "Classic Backward"]]
print_Runtime(t_list_1, 'top_right')
t_list_1.append([time_forward, "Forward Method"])
print_Runtime(t_list_1, 'top_right')
# myprint.print_Utility(True, weekly_Backward_h, gamma, 0, 0.5, 0.001)
# myprint.print_Utility(False, weekly_Forward_h_cum, gamma, 0, 0.5, 0.001)
e_list = [[earning_b, "Backward with update"], [earning, "Classic Backward"]]
e_list = [[earning_b, "Backward with update"], [earning, "Classic Backward"],[earning_f, "Forward Method"]]
# myprint.print_Earning(e_list)

#myprint.print_Utility(False, weekly_Forward_h_cum, gamma, 0, 0.5, 0.001)


'''
plt.plot(range(len(all_Close_prices)), all_Close_prices)
plt.show()

plt.plot(range(len(prices)), prices)
plt.show()
'''

'''
visualization
'''



'''
debug
'''

#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# fig = plt.figure()
# ax1 = fig.add_subplot(3, 1, 1)
# ax2 = fig.add_subplot(3, 1, 2)
# ax3 = fig.add_subplot(3, 1, 3)
# ax1.set_ylabel('Stock Price')
# ax2.set_ylabel('Backward Alpha')
# ax3.set_ylabel('Forward Alpha')
#
# y = all_Close_prices
# x = [i for i in range(40)]
# y_b = [0]*40 + list(weekly_Backward_alpha)
# y_f = [0]*40 + list(weekly_Forward_alpha)
# stock, = ax1.plot(x, [min(y), max(y)]*20)
# ax2.set_ylim(0, 1e-3)
# ax3.set_ylim(0, 1e-3)
# back, = ax2.plot(x, [min(y_b), max(y_b)]*20)
# forward, = ax3.plot(x, [min(y_f), max(y_f)]*20)
#
# def init():
#     stock.set_ydata([0] * len(x))
#     back.set_ydata([0] * len(x))
#     forward.set_ydata([0] * len(x))
#     return stock,back, forward
#
# def animate(i):
#     stock.set_ydata(y[i:i+40]) # update the data.
#     back.set_ydata(y_b[i:i+40])  # update the data.
#     forward.set_ydata(y_f[i:i+40])  # update the data.
#     return stock, back, forward
#
#
# ani = animation.FuncAnimation(
#     fig, animate, init_func=init, interval=300, blit=True, save_count=10)
#
# ani.save("movie.mp4")
# # plt.show()




