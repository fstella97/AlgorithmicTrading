# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:00:04 2020

@author: Francesco Stella 
"""
def daily_returns(prices):
    res = (prices/prices.shift(1) - 1.0)[1:]
    return res


def cumulative_returns(returns):
    res = (returns + 1.0).cumprod()
    return res

def lin_regress(x,y):
        X = x
        Y = y
        X = sm.add_constant(X) # adding a constant
        model = sm.OLS(Y, X)
        p = model.fit().params
        return p

def plot_figure2(data1,data2,xaxis,yaxis,title):
  plt.figure()
  plt.style.use('seaborn-paper')  #by changing the style the result change dramatically
  # some good examples are 'fivethirtyeight', 'seaborn-dark', 'seaborn-paper','classic','default'
  plt.plot(data1,linewidth=1.5,label='PEP')
  plt.plot(data2,linewidth=1.5,label='PLUG') #other options are 'bs' 'g^'
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
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import datetime 
#from scipy.optimize import minimize
import yfinance as yf
import statsmodels.api as sm

# Specifying parameters
stocks = ['BTC-USD','SPY','ASML','SPWR','ECAR.L','XLC','FDIS','EWL','INDA'] #'GRWG','NIO','JPM','C','PLUG','SAF','AMZN','PEP','AIR','ASML','SPWR','NIO','ECAR.L','GRWG',Stocks in consideration , other possible pairs ['FDX','UPS']  ['KO','PEP']  ['JPM','C'] ['NFLX','AMZN']
start = '2020-11-20'  #Start of the data
end =   '2021-10-20'  #End of the data
fee = 0.001           #Fees from the broker 

#Getting data from Yahoo Finance
data = pd.DataFrame()
stddev = pd.DataFrame()
returns = pd.DataFrame()
percentage = pd.DataFrame()
lin= pd.DataFrame()
tot_prices=pd.DataFrame()
plt.figure(1)
for stock in stocks:
        prices = yf.download(stock, start, end)
        data[stock] = prices['Adj Close'] # According to the book you have to use Adj close, while the youtube guy uses Close
        tot_prices=pd.concat([tot_prices, data[stock]], axis=1)
        percentage[stock]=data[stock]/data[stock][0]*100
        returns[stock] = np.append(data[stock][1:].reset_index(drop=True)/data[stock][:-1].reset_index(drop=True) - 1 , 0)
        stddev[stock]=data[stock].std()
        days_of_trading=np.linspace(0, len(data[stock]), len(data[stock]), endpoint = False)
        p=lin_regress(days_of_trading,percentage[stock])
        lin[stock]=p
        plt.style.use('fivethirtyeight')
        plt.plot(days_of_trading, p.const + p[1] * days_of_trading)
        plt.plot(days_of_trading, percentage[stock],label=stock)
        plt.title(stock,fontsize=18, color='k') 
        plt.legend(loc='upper right')
        plt.grid(b=True, which='major')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()

d_returns=daily_returns(tot_prices)
cum_returns = cumulative_returns(daily_returns(tot_prices))
cov = cum_returns.cov()        
meand_return=d_returns.mean()

plt.figure(figsize=(15,10))
for stock in stocks:
    plt.scatter(cov[stock][stock],meand_return[stock],s=150,alpha=0.8,label=stock)
    plt.text(cov[stock][stock]+0.01, meand_return[stock]-0.00006, stock, fontsize=20)
plt.style.use('seaborn-dark')
plt.title('Stocks comparison',fontsize=38, color='k') 
legend =plt.legend(loc='lower right',fontsize=17,frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('k')
plt.grid(b=True, which='major')
plt.minorticks_on()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Std',fontsize=30)
plt.ylabel('Mean daily return',fontsize=30)
#plt.xlim(-0.05, 0.4)
#plt.ylim(-0.0005, 0.003)
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()    