import numpy as np
import pandas as pd



def get_h_helper(q, pu, pd):
    '''
    helper function
    '''
    return q*np.log(q/pu) + (1-q)*np.log((1-q)/pd)

def solve_Backward(period, market_model, S0, gamma):
    '''
    Solve Backward Preference
    @param period:
    @param market_model: a list contains [u, d, pu, pd]
    @param S0: the initial stock price
    @param gamma: a pre-assumed constant that represents the degree of risk preference of the utility function at Terminal time
    @return: a list contains h_0 (information related to utility function at time 0) and alpha0 (the initial investment)
    '''
    u = market_model[0]
    d = market_model[1]
    pu = market_model[2]
    pd = market_model[3]
    q = (1-d)/(u-d)
    if period == 1:
        h_0 = get_h_helper(q, pu, pd)
        alpha0 = (np.log(pu / q) - np.log(pd / q)) / (gamma * S0 * (u - d))
        return [h_0, alpha0]
    h_t = np.zeros(period)
    for i in np.arange(period-1, 1, -1):
        h_t_1 = np.zeros(i)
        for j in range(i):
            h = get_h_helper(q, pu, pd)
            pre_h_u = h_t[j]
            pre_h_d = h_t[j+1]
            h += q * pre_h_u + (1-q) * pre_h_d
            h_t_1[j] = h
        h_t = h_t_1
    h_1 = h_t
    h_0 = get_h_helper(q, pu, pd) + q * h_1[0] + (1-q) * h_1[1]
    alpha0 = (np.log(pu/q) - np.log(pd/q) - (h_1[0]-h_t[1])) / (gamma*S0*(u-d))
    return [h_0, alpha0]

def solve_Forward(market_model, S0, gamma):
    '''
    Solve Forward Preference
    @param market_model: a list contains [u, d, pu, pd]
    @param S0: the initial stock price
    @param gamma: a pre-assumed constant that represents the degree of risk preference of the utility function at Terminal time
    @return: a list contains h_0 (information related to utility function at time 0) and alpha0 (the initial investment)
    '''
    u = market_model[0]
    d = market_model[1]
    pu = market_model[2]
    pd = market_model[3]
    q = (1-d)/(u-d)
    h = -get_h_helper(q, pu, pd)
    alpha0 = np.log((u-1)*pu/((1-d)*pd)) / (gamma*S0*(u-d))
    return [h, alpha0]