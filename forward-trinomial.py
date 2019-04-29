''''

Forward Multi-Assets

'''
import numpy as np
import pandas as pd
from itertools import product
import scipy.optimize as opt
import myprint
import ds

all_stock = pd.read_csv("./data/GSPC.csv")
prices = np.array(all_stock['Close'].values)
all_Close_prices = prices[2::5]

def get_pu_tri(prices):
    '''
    Using Monte Carlo method to estimate pu and pd
    @return: a list contains pu and pd
    '''
    pu = 0
    pd = 0
    for i in range(len(prices)-1):
        if prices[i] < prices[i+1]:
            pu += 1
    pu = pu / len(prices)
    return [pu, 1-pu]

def get_u():
    return