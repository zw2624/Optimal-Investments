import pandas as pd
import numpy as np
import pandas as pd
import datetime
import xlrd

data_dir = './data/'

def get_CloseFromRaw(name):
    raw = pd.read_csv(data_dir + name + '.csv')
    start = next(i for i in range(len(raw['Date'])) if datetime.datetime.strptime(raw['Date'][i], '%Y-%m-%d').strftime('%A') == 'Friday')
    weekly_close = get_ClostPrices(raw['Adj Close'], start, 5)
    return weekly_close


def get_model(price):
    w = 5
    para = get_parameters(price, w)
    sigma = para[1]
    tau = w / 365
    u = np.exp(sigma*np.sqrt(tau))
    d = 1 / u
    pu = get_pu_MC(price)[0]
    pd = get_pu_MC(price)[1]
    return [u-1, d-1, pu, pd]


def get_ClostPrices(prices, start, w):
    return prices[start::w]


def get_parameters(close_prices, w):
    '''
    Get the market estimates.
    @param prices: stock prices, np-array
    @param start: index of first Close Price
    @param w: window between two Close Price (5 if it is weekly)
    @return: a list contains volatility, sigma, and drift
    '''
    tau = w / 365
    ui = np.diff(np.log(close_prices)) # ui is the log of the weekly return
    n = ui.shape[0]
    ubar = ui.mean()
    volatility = np.sqrt(sum((ui - ubar) ** 2) / (n - 1))
    sigma = volatility / tau**0.5
    return [volatility, sigma]


def get_discounted(prices, rate):
    '''
    Discount the prices series
    @param prices: stock prices
    @param rate: discount rate + 1
    @return: discounted prices
    '''
    return prices / [rate ** i for i in range(len(prices))]


def get_market_model(sigma, w, useMC, prices = None):
    '''
    Get the parameters for binomial model
    @param sigma:
    @param w: window
    @param useMC: if using Monte Carlo Method to generate pu, pd.
    @return: a list contains u, d, pu, pd
    '''
    tau = w / 365
    u = np.exp(sigma*np.sqrt(tau))
    d = 1 / u
    if useMC:
        pu = get_pu_MC(prices)[0]
        pd = get_pu_MC(prices)[1]
    else:
        pu = (1 - d)/ (u - d)
        pd = 1 - pu
    return [u-1, d-1, pu, pd]


def get_tri_market_model(sigma, w, lam):
    '''
    :param lam >= 1
    '''
    tau = w / 365
    u = np.exp(sigma * np.sqrt(tau))
    V = lam * sigma * np.sqrt(tau)
    p1 = 1 / (2*lam**2) + u * np.sqrt(tau) / (2*lam*sigma)
    p2 = 1 - 1 / lam**2
    p3 = 1 - p1 - p2
    return [V, p1, p2, p3]


def get_pu_MC(prices):
    '''
    Using Monte Carlo method to estimate pu and pd
    @return: a list contains pu and pd
    '''
    pu = 0
    for i in range(len(prices)-1):
        if prices[i] < prices[i+1]:
            pu += 1
    pu = pu / (len(prices)-1)
    return [pu, 1-pu]