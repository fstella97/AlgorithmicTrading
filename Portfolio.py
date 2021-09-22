# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:00:04 2020

@author: juven
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datetime 
import quandl 
from scipy.optimize import minimize
import pandas_datareader as pdr


plug = pdr.get_data_yahoo('PLUG', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))
plug=plug['Close'][:]

"asan=pdr.get_data_yahoo('ASAN', start=datetime.datetime(2020, 10, 2),end=datetime.datetime(2020, 10, 20))"
safr=pdr.get_data_yahoo('SAF', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))

safr=safr['Close'][:]
#veur= pdr.get_data_yahoo('VEUR', start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2020,10, 20))

amzn=pdr.get_data_yahoo('AMZN', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))
amzn=amzn['Close'][:]

inrg=pdr.get_data_yahoo('INRG', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))
inrg=inrg['Close'][:]

air=pdr.get_data_yahoo('AIR', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))
air=air['Close'][:]

asml=pdr.get_data_yahoo('ASML', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))
asml=asml['Close'][:]

tsla=pdr.get_data_yahoo('TSLA', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))
tsla=tsla['Close'][:]

vow3=pdr.get_data_yahoo('VOW3.DE', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))
vow3=vow3['Close'][:]


spwr = pdr.get_data_yahoo('SPWR', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))

spwr=spwr['Close'][:]

nio = pdr.get_data_yahoo('NIO', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))

nio=nio['Close'][:]

ecar = pdr.get_data_yahoo('ECAR.L', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))

ecar=ecar['Close'][:]

grwg = pdr.get_data_yahoo('GRWG', 
                          start=datetime.datetime(2006, 1, 1), 
                          end=datetime.datetime(2020, 12, 20))

grwg=grwg['Close'][:]


stocks=pd.concat([plug, safr, amzn, inrg, asml, tsla, ecar, nio, spwr],axis=1)
stocks.columns=['plug', 'safr','amzn','inrg','asml', 'tsla','ecar','nio','spwr']

#stocks=pd.concat([plug, amzn, tsla],axis=1)
#stocks.columns=['plug','amzn','tsla']
print(stocks.head())

log_ret=np.log(stocks/stocks.shift(1))
print(log_ret.head())

np.random.seed(42)
num_ports = 6001
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports-1):
    # Weights
    weights = np.array(np.random.random(len(stocks.columns)))
    weights = weights/np.sum(weights)
    
    # Save weights
    all_weights[x,:] = weights
    
    # Expected return
    ret_arr[x] = np.sum( (log_ret.mean() * weights * 252))
    
    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    
    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]


# Weights
weights_c = np.array(np.ones(len(stocks.columns)))
weights_c = weights_c/np.sum(weights_c)

# Save weights
# all_weights[6000,:] = weights_c

# Expected return
# ret_arr[6000] = np.sum( (log_ret.mean() * weights_c * 252))

# Expected volatility
# vol_arr[6000] = np.sqrt(np.dot(weights_c.T, np.dot(log_ret.cov()*252, weights_c)))

# Sharpe Ratio
# sharpe_arr[6000] = ret_arr[6000]/vol_arr[6000]
# current_sr_ret=ret_arr[6000]
# current_sr_vol=vol_arr[6000]

print('Max sharpe ratio in the array:{}'.format(sharpe_arr.max()))
print('Its location in the array:{}'.format(sharpe_arr.argmax()))

print(['plug','safr','amzn','inrg','asml', 'tsla','ecar','nio','spwr'])
print(all_weights[sharpe_arr.argmax(),:])
max_sr_ret=ret_arr[sharpe_arr.argmax()]
max_sr_vol=vol_arr[sharpe_arr.argmax()]
    
plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot
# plt.scatter(current_sr_vol, current_sr_ret,c='black', s=50) # red dot
plt.show()


print('Investment per stock:{}' .format(4000*all_weights[sharpe_arr.argmax(),:]))

def get_ret_vol_sr(weights):
    #weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) 
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov(), weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

def neg_sharpe(weights):
# the number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    #return 0 if sum of the weights is 1
    return np.sum(weights)-1


cons=({'type':'eq', 'fun':check_sum})
bounds=((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
#bounds=((0,1),(0,1),(0,1))
init_guess=(1/len(stocks.columns))*np.ones(len(stocks.columns))

opt_results=minimize(neg_sharpe, init_guess, method='SLSQP',bounds=bounds,constraints=cons)
print(opt_results)

get_ret_vol_sr(opt_results.x)
print(opt_results.x)

frontier_y=np.linspace(0,0.3,200)

def minimize_volatility(weights):
        return get_ret_vol_sr(weights)[1]


frontier_x = []

for possible_return in frontier_y:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = minimize(minimize_volatility,init_guess,method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])
    
# Short moving window rolling mean
aapl['42'] = adj_close_px.rolling(window=40).mean()

# Long moving window rolling mean
aapl['252'] = adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
aapl[['Adj Close', '42', '252']].plot()

# Show plot
plt.show()
    
plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)
plt.savefig('cover.png')
plt.show()

