#!/usr/bin/env python
# coding: utf-8

# In[1003]:


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# In[1098]:


class FinancialInstrumentBase(): # parent class
    '''Class to analyze financial instruments like stocks. Data downloaded from Yahoo Finance'''
    def __init__(self, ticker, start, end, normal = False):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.normal = normal
        self.get_data()
        self.log_returns()
    
    def __repr__(self): #use this dunder method to document the class structure
        return "FinancialInstrument(ticker = {}, start = {}, end = {}, Normalized = {})".format(self._ticker, 
                                                                                                self.start, 
                                                                                                self.end, 
                                                                                                self.normal)
    
    def get_data(self):
        self.data = yf.download(self._ticker, self.start, self.end).Close.to_frame()
        self.data.rename(columns = {"Close":"price"}, inplace = True)
        if self.normal:
            self.data["price"] = self.data.price.div(self.data.price.iloc[0]).mul(100).copy() #normalize
        
    
    def log_returns(self):
        self.data["log_returns"] = np.log(self.data.price/self.data.price.shift(1)).dropna()
    
    def set_ticker(self, ticker = None):
        if ticker is not None:
            self._ticker = ticker
            self.get_data()
            self.log_returns()
    
    def plot_prices(self):
        '''Plots both time series and frequency of returns''' #docstring
        #show historical price action
        fig1 = plt.figure(figsize=(15,8))
        plt.plot(self.data.price)
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("Price, $ USD", fontsize=15)
        plt.suptitle("Historical Price Chart for {} from {} to {}".format(self._ticker, self.start, self.end), fontsize=15)
        plt.grid(which="both")
        plt.show()
        
    def plot_returns(self): #show daily log returns
        self.data.log_returns.plot(figsize=(15,8))
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("% Returns (Log)", fontsize=15)
        plt.suptitle("Daily Log Returns for {} from {} to {}".format(self._ticker, self.start, self.end), fontsize=15)
        plt.grid(which="both")        
        plt.show()
        
        #show daily log returns distribution
        stock.data.log_returns.plot(kind="hist", figsize=(15,8), bins=int(np.sqrt(len(self.data))))
        plt.xlabel("% Returns (Log)", fontsize=15)
        plt.ylabel("Number of Days", fontsize=15)
        plt.suptitle("Frequency of Returns for {} from {} to {}".format(self._ticker, self.start, self.end), fontsize=15)        
        plt.grid(which="both")        
        plt.show()


# In[1103]:


class RiskReturn(FinancialInstrumentBase): #child class
    def __repr__(self): #use this dunder method to document the class structure
        return "RiskReturn(ticker = {}, start = {}, end = {}, Normalized = {})".format(self._ticker, 
                                                                                       self.start, 
                                                                                       self.end, 
                                                                                       self.normal)
    
    def __init__(self, ticker, start, end, normal = False, freq = None):
        self.freq = freq
        super().__init__(ticker, start, end, normal = False)
        
    def mean_return(self, freq = None):
        if freq is None:
            return self.data.log_returns.mean()
        else:
            resampled_price = self.data.price.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()
    
    def std_returns(self, freq = None):
        if freq is None:
            return self.data.log_returns.std()
        else:
            resampled_price = self.data.price.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()
        
    def annualized_perf(self):
        mean_return = round(self.data.log_returns.mean() * 252, 3)
        risk = round(self.data.log_returns.std() * np.sqrt(252), 3)
        return "Annualized Performance for {}: Return: {} | Risk {}".format(self._ticker, mean_return, risk)


# In[1104]:


stock = RiskReturn("AAPL", "2017-01-01", "2019-1-31", False)


# In[1105]:


stock


# In[1106]:


stock._ticker


# In[1085]:


stock.start


# In[1086]:


stock.end


# In[1087]:


stock.data


# In[1088]:


stock.mean_return("BY") #business year


# In[1089]:


stock.std_returns("BY")


# In[1090]:


stock.mean_return("W-Fri")


# In[1091]:


stock.annualized_perf()


# In[1092]:


stock.plot_prices()


# In[1093]:


stock.plot_returns()


# In[1094]:


stock.data


# In[1095]:


stock.annualized_perf()


# In[1096]:


#round(stock.data.price.loc['2015-12-30'],3)


# In[1097]:


stock.mean_return("D")


# In[ ]:




