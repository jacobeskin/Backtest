"""
A code for downloading time series data and running a backtest of a very 
simplistic trading strategy. The strategy is buy/sell when certain conditions 
the price movement and trading volume are met. The focus of the code is the 
backtesting class.
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as dt
import matplotlib.pyplot as plt


# The method for fetching data from Yahoo finance. In order to use in the
# backtesting class below, get enough data, over 51 days.
def getData(start,end):
    """
    Inputs are the start date and end date of the time series data in the format
    'YYYY-MM-DD'. Output is a dictionary of pandas dataframes where the key is
    the ticker symbol and value is a dataframe. Uses first 50 days of the data
    to calculate some averages, so there shoud be enough data to calculate them
    and do the backtest of the amount of days one wants. Instrument tickers are 
    read from a pre defined .csv file.
    """

    # Define a few variables for error reporting
    error = 0
    data_missing = 0

    # Define the output dictionary and fetch data of the SPDR S&P 500 ETF
    data = {}
    data['SPY'] = dt.DataReader('SPY', 'google', start, end)

    # Get (or try to get) the data for all the stocks that make up S&P 500 index
    print('\n', 'Fetching the data...')

    tckrs = pd.read_csv('constituents_csv.csv')
    for i in range(len(tckrs)):
        # Get the data, check if its the same length of SPY and if so
        # save it to dictionary, otherwise count up the errors
        try:
            d = dt.DataReader(tckrs.iloc[i][0], 'google', start, end)
            if len(d)==len(data['SPY']): data[tckrs.iloc[i][0]] = d
            else: data_missing += 1
        except: # Might be risky to not incule error categories...
            error += 1

    # Print the numer of errors
    print(' Number of instruments dropped due to error: ', error)
    print(' Number of instruments dropped due to missing data: ', data_missing)
    print(' Number of instruments succesfully fetched: ', len(data))

    # Return the data
    return(data)

# The class that actually does the backtesting
class Backtest:
    """
    Class that perfoms the backtesting of the simple strategy. Constructor
    takes in just the data dictionary given out by the getData() method. The 
    backtest is done buy running through the dataset one day at a time
    making decisions on opening, closing or adjusting positions for each 
    stock on that day.
    """

    # Constructor
    def __init__(self, data):
        self.data = data

        # Initialize the position dictionary
        self.pos = {}

        # These are for keeping record of profits and losses
        self.long_PL = []    # PnL from closed long trades
        self.short_PL = []   # PnL from closed short trades
        self.total_c_PL = [] # Total PnL from closed positions
        self.total_portfolio_value_l = [] # PnL of open long positions
        self.total_portfolio_value_s = [] # PnL of open short positions
        self.total_portfolio_value = []   # Total PnL for open positions
        self.total_PnL = []               # Total PnL
        
        # For keeping track of number of trades
        self.n_trades_l = 0 # number of long trades
        self.n_trades_s = 0 # number of short trades
        self.lPL = 0
        self.sPL = 0

    # ---------- Indicators for opening positions ----------

    # Indicator following the ETF, input k is indexing the date
    def ind_SPY(self, k):

        # Calculate the average one day momentum from previous 50 days
        prices1 = np.array(self.data['SPY']['Close'][k-51:k])
        prices2 = np.array(self.data['SPY']['Close'][k-50:k+1])
        mom = prices2-prices1
        ave_mom = np.mean(np.absolute(mom))

        # Calculate the average one day volume from previous 50 days
        vol = np.array(self.data['SPY']['Volume'][k-50:k+1])
        ave_vol = np.mean(vol)

        # Calculate the moving averages of the prices
        
        # 10 day moving average
        m_ave10 = np.mean(np.array(self.data['SPY']['Close'][k-10:k+1]))
        m_ave10_prev = np.mean(np.array(self.data['SPY']['Close'][k-11:k]))
        
        # 50 day moving average
        m_ave50 = np.mean(np.array(self.data['SPY']['Close'][k-50:k+1]))

        # The change in the 10 day moving average
        d_m_ave = m_ave10-m_ave10_prev

        # Determining the trading signal (buy, sell, do nothing....)
        if ((mom[-1]<0)and(np.absolute(mom[-1])>ave_mom)and(vol[-1]>ave_vol)and
            (d_m_ave<0)and(m_ave10<m_ave50)):
            return('SELL') # Sell (or short)
        elif ((mom[-1]>0)and(np.absolute(mom[-1])>ave_mom)and(vol[-1]>ave_vol)
              and(d_m_ave>0)and(m_ave10>m_ave50)):
            return('BUY')  # Buy
        else: return(0)     # i.e. do nothing


    # Indicator following the different stocks
    def ind(self, ticker, k):
 
        # Calculate the average one day momentum from previous 50 days
        prices1 = np.array(self.data[ticker]['Close'][k-51:k])
        prices2 = np.array(self.data[ticker]['Close'][k-50:k+1])
        mom = prices2-prices1
        ave_mom = np.mean(np.absolute(mom))

        # Calculate the average one day volume from previous 50 days
        vol = np.array(self.data[ticker]['Volume'][k-50:k+1])
        ave_vol = np.mean(vol)

        # Calculate the moving averages of the prices
        m_ave5 = np.mean(np.array(self.data[ticker]['Close'][k-5:k+1]))
        m_ave20 = np.mean(np.array(self.data[ticker]['Close'][k-20:k+1]))

        # Determining the trading signal (buy, sell, do nothing....)
        if ((mom[-1]<0)and(np.absolute(mom[-1])>ave_mom)and(vol[-1]>ave_vol)and
            (m_ave5<m_ave20)):
            return('SELL') # Sell (or short)
        elif ((mom[-1]>0)and(np.absolute(mom[-1])>ave_mom)and(vol[-1]>ave_vol)
              and(m_ave5>m_ave20)):
            return('BUY')  # Buy
        else: return(0)     # i.e. do nothing

    #---------------------------------------------------------------------------

    # Method for adjusting the positions, k is numbering the observation and
    # spy_ind is the output of ind_spy()
    def adj_pos(self, ticker, k, spy_ind):

        # Current price and indicator signal
        price = self.data[ticker]['Close'][k]
        ind = self.ind(ticker, k)

        # Adjusting position, position information is kept in a dictionary
        # self.pos{} that is initialized in the constructor. Logs of trades
        # trades are recorded in text files called self.log_l and self.log_s.
        # The cumulative PnL is also updated in self.lPl and self.sPl of long
        # and short positions respectively. self.last is for the last trading
        # day and all positions are closed. self.portval_l and self.portval_s
        # are for recording the value of the open positions.

        # If there is an open position with this ticker
        if ticker in self.pos:

            # Closing long position
            if (((self.pos[ticker][0]=='BUY')and(price<self.pos[ticker][2]))or
                ((self.pos[ticker][0]=='BUY')and self.last==True)):
                p_l = price-self.pos[ticker][1] # P/L of this position
                self.lPL += p_l                 # Update cumulative PnL
                del self.pos[ticker]            # Open position deleted
                # Write this in the log
                _ = self.log_l.write('Close long position in '+ticker+' at '
                                     + str(price)+' with PL '+str(p_l)+'.\n')

            # Closing short position
            elif (((self.pos[ticker][0]=='SELL')and(price>self.pos[ticker][2]))
                  or((self.pos[ticker][0]=='SELL')and self.last==True)):
                p_l = self.pos[ticker][1]-price
                self.sPL += p_l
                del self.pos[ticker]
                # Write into the log
                _ = self.log_s.write('Close short position in '+ticker+' at '
                                     + str(price)+' with PL '+str(p_l)+'.\n')

            # Adjusting stop loss and/or recording the current value of the
            # position. stop loss is 5% from the current level if current level
            # is better than the opening price (in the beginning it's 10%)  
            else: 
                if (self.pos[ticker][0]=='BUY'):
                    self.portval_l += price-self.pos[ticker][1]
                    if (price>=self.pos[ticker][3]):
                        self.pos[ticker][2] = price-0.05*price
                        self.pos[ticker][3] = price
                    
                elif (self.pos[ticker][0]=='SELL'):
                    self.portval_s += self.pos[ticker][1]-price
                    if (price<=self.pos[ticker][3]):
                        self.pos[ticker][2] = price+0.05*price
                        self.pos[ticker][3] = price

        # Opening positions. The position data is recorded in the self.pos{}
        # dictionary, where the key is again the instrument ticker and the
        # return value is a list of information of the position. The first
        # element of the list is the position type (buy/sell), the second is
        # the starting price of the position, the third is the stop loss
        # price and the fourth is current price or the previous days price.
        # self.n_trades_l and self.n_trades_s keep count of number of trades

        if ticker not in self.pos:

            # Open a long position        
            if (ind=='BUY') and (spy_ind=='BUY')and(self.last==False):
                pos_start = price
                stop_loss = price-0.1*price
                self.pos[ticker] = ['BUY', pos_start, stop_loss, price]
                # Write this into the log
                _ = self.log_l.write('Open long position in '+ticker+' at '
                                     +str(pos_start)+'.\n')
                self.n_trades_l += 1

            # Open a short position
            if (ind=='SELL')and(spy_ind=='SELL')and(self.last==False):
                pos_start = price
                stop_loss = price+0.01*price
                self.pos[ticker] = ['SELL', pos_start, stop_loss, price]
                _ = self.log_s.write('Open short position in '+ticker+' at '
                                     +str(pos_start)+'.\n')
                self.n_trades_s += 1


    # ---------- The actual backtesting method ----------

   #Takes in the number of days to backtest as a variable.

    def backTest(self, backtest_days):

        # Logs and last trading day signal
        self.log_s = open('BacktestLog_short.txt','w') # log of all short trades
        self.log_l = open('BacktestLog_long.txt', 'w') # log of all long trades
        self.last = False
        
        # Start going through the data
        for i in range(len(self.data['SPY'])-
                       backtest_days,len(self.data['SPY'])):

            # If the last day of backtest is reached
            if i==len(self.data['SPY'])-1: self.last = True

            # Update the dates on the logs
            _ = self.log_l.write('\n'+str(self.data['SPY'].index.values[i])+
                                 ':\n')
            _ = self.log_s.write('\n'+str(self.data['SPY'].index.values[i])+
                                 ':\n')

            # more variables for keeping track of and calculating the PnL.
            tPL = 0
            self.portval_l = 0 # Current PnL of open long positions
            self.portval_s = 0 # Current PnL of open short positions

            # Calcuating the ETF indicator value
            spy_ind = self.ind_SPY(i)

            # Going through the instruments
            for ticker in self.data:
                self.adj_pos(ticker, i, spy_ind)
                tPL = self.lPL+self.sPL

            # ....more about PnL, for plotting....
            self.long_PL.append(self.lPL)
            self.short_PL.append(self.sPL)
            self.total_c_PL.append(tPL)
            self.total_portfolio_value_l.append(self.portval_l)
            self.total_portfolio_value_s.append(self.portval_s)
            self.total_portfolio_value.append(self.portval_l+self.portval_s)
            self.total_PnL.append(self.total_portfolio_value[-1]+
                                  self.total_c_PL[-1])
        # Close logs
        self.log_l.close()
        self.log_s.close()

#-------------------------------------------------------------------------------

# Next a method for testing out the above code. The test method fetches the data
# and runs the backtest and plots the PnL
def test():

    # Get data
    data = getData('2016-12-19', '2017-12-18')

    # Backtest strategy
    l = 200
    strategy = Backtest(data)
    strategy.backTest(l)

    # Print some results
    print('\n', 'Number of long trades: ', strategy.n_trades_l)
    print(' Number of short trades: ', strategy.n_trades_s)
    print(' Total PnL: ', strategy.total_c_PL[-1], '\n')

    # Make a few plots
    # x-axis for the plot
    x = len(data['SPY'])-l
    x_axis = np.array(pd.Series(data['SPY'].index.values[x:]).dt.date)

    # Plot PnL
    f = plt.figure(1)
    plt.axis([x_axis[0], x_axis[-1], -600, 2500])
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.plot(x_axis, strategy.total_c_PL, 'b', label='PnL of closed positions')
    plt.plot(x_axis, strategy.total_portfolio_value, 'r',
             label='PnL of open positions')
    plt.plot(x_axis, strategy.total_PnL, 'g', label='Total PnL')
    plt.axhline(0, color='black')
    plt.legend(loc='upper left', ncol=1)
    plt.title('PnL')
    f.show()
    
            
            



                    


            

        
        

        

        
        
       


    
    
    

