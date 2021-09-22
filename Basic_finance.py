# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:53:33 2021

@author: Francesco Stella 
 Basic algorithmic trading
"""

""" First option to import stock data"""
import pandas as pd
import pandas_datareader as pdr 
import datetime

import plotly
import plotly.offline as py
import plotly.graph_objs as go
"""py.init_notebook_mode(connected=True)"""


btc=pdr.get_data_yahoo('BTC',
                        start=datetime.datetime(2020,10,1),
                        end=datetime.datetime(2021,1,1))

""" Second option to import stock data as DataFrame"""
import quandl 
aapl=quandl.get("WIKI/AAPL",start_date="2006-10-01",end_date="2012-01-01")

" get the firts or last rows of data, or describe it" 
aapl.head()
aapl.tail()
print(aapl.describe())

print(aapl)










""" To create a candlestick plot of it 


plotly.io.renderers.default = 'browser'
start = datetime.datetime(2020,2,19)
end = datetime.datetime(2021,5,30)
df = pdr.DataReader('BTC-USD','yahoo',start,end)

data = [go.Candlestick(x=df.index,
                       open=df.Open,
                       high=df.High,
                       low=df.Low,
                       close=df.Close)]

layout = go.Layout(title='Bitcoin Candlestick with Range Slider',
                   xaxis={'rangeslider':{'visible':True}})

fig = go.Figure(data=data,layout=layout)
py.iplot(fig,filename='bitcoin_candlestick')
xaxis = {'rangeselector':{'buttons':[{'count':1,
                                      'label':'1m',
                                      'step':'month',
                                      'stepmode':'backward'}]}}
"""