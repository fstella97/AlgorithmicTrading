"""
Created on Thu Jun  3 21:11:17 2021

@author: Francesco Stella 
"""
"""https://www.youtube.com/watch?v=jvZ0vuC9oJk"""
"""Basic idea: PAIR TRADING:
    Go short and long at the some time on two highly correlated stoks, when they deviate from their correlation, assuming that the mispricing would close down. 
First, find two higly correlated equities (I think it would be over 0.8, but let s check it later).


"""
#Importing packages
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt

# Specifying parameters
stocks = ['KO','PEP'] #Stocks in consideration , other possible pairs ['FDX','UPS']  ['KO','PEP']  ['JPM','C'] ['NFLX','AMZN']
start = '2019-12-31' #Start of the data
end =   '2021-06-03'   #End of the data
fee = 0.001          #Fees from the broker 
window = 252         #Window for the cointegration parameter estimation (1 year=252 trading days)
t_threshold = -2.5   #Threshold related to the reliability of the cointegration, the smaller the better

#Getting data from Yahoo Finance
data = pd.DataFrame()
returns = pd.DataFrame()
for stock in stocks:
        prices = yf.download(stock, start, end)
        data[stock] = prices['Adj Close'] # According to the book you have to use Adj close, while the youtube guy uses Close
        returns[stock] = np.append(data[stock][1:].reset_index(drop=True)/data[stock][:-1].reset_index(drop=True) - 1 , 0)
 
#initialising arrays
gross_returns = np.array([])
net_returns = np.array([])
t_s = np.array([])  #for optimal cointegration statistic (is the pair becoming more or less cointegrated over time?)
stock1 = stocks[0]
stock2 = stocks[1]
#moving through the sample
for t in range(window, len(data)):#from start to window for cointegration, for window to end to trade
    #defining the unit root function: stock2 = a + b*stock1
    def unit_root(b):
       a = np.average(data[stock2][t-window:t] - b*data[stock1][t-window:t])
       fair_value = a + b*data[stock1][t-window:t]
       diff = np.array(fair_value - data[stock2][t-window:t])#to estimate which stock is overvalued
       diff_diff = diff[1:] - diff[:-1]#to see if the difference is converging to 0 (desired) or not
       reg = sm.OLS(diff_diff, diff[:-1])
       res = reg.fit()
       return res.params[0]/res.bse[0]
    #optimising the cointegration equation parameters
    res1 = spop.minimize(unit_root, data[stock2][t]/data[stock1][t], method='Nelder-Mead')#optimizer (function,initpoint, method)
    t_opt = res1.fun      # the value of the function from the optimization
    b_opt = float(res1.x) #each day we recompute the b value given past data
    a_opt = np.average(data[stock2][t-window:t] - b_opt*data[stock1][t-window:t])#each day we recompute the a value given past data
    #simulating trading
    fair_value = a_opt + b_opt*data[stock1][t]
    if t == window:
         old_signal = 0
    if t_opt > t_threshold: #the cointegration is not reliable enough to trade 
         signal = 0
         gross_return = 0
    else:
        signal = np.sign(fair_value - data[stock2][t])
        gross_return = signal*returns[stock2][t] - signal*returns[stock1][t]
    fees = fee*abs(signal - old_signal)
    net_return = gross_return - fees
    gross_returns = np.append(gross_returns, gross_return)
    net_returns = np.append(net_returns, net_return)
    t_s = np.append(t_s, t_opt)
    # interface printing positions and returns of the simulated trades
    print('day '+str(data.index[t]))
    print('')
    if signal == 0:
       print('no trading')
    elif  signal == 1:
       print('long position on '+stock2+' and short position on '+stock1)
    else:
       print('long position on '+stock1+' and short position on '+stock2)
    print('gross daily return: '+str(round(gross_return*100,2))+'%')
    print('net daily return: '  +str(round(net_return*100,2))+'%')
    print('cumulative net return so far: '+str(round(np.prod(1+net_returns)*100-100,2))+'%')
    print('')
    old_signal = signal
    
   
def plot_figure2(data1,data2,xaxis,yaxis,title):
  plt.figure()
  plt.style.use('seaborn-paper')  #by changing the style the result change dramatically
  # some good examples are 'fivethirtyeight', 'seaborn-dark', 'seaborn-paper','classic','default'
  plt.plot(data1,linewidth=1.5,label='Gross return')
  plt.plot(data2,linewidth=1.5,label='Net Returns') #other options are 'bs' 'g^'
  plt.title(stocks,fontsize=18, color='k') 
  plt.xlabel(xaxis)
  plt.ylabel(yaxis)
  #plt.axis([0,6,0,10]) #Limits of the axes, first x and then y 
  plt.legend(loc='upper right')
  plt.grid(b=True, which='major')
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
  plt.show()
  return 'plot eseguito'
    
a= plot_figure2((np.append(1,np.cumprod(1+gross_returns))*100),(np.append(1,np.cumprod(1+net_returns))*100),'Trading days','Percentual result',stocks)
 