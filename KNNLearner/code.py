######################################################################################################
# Mark Trinquero
# Machine Learning for Trading
# Project 3.2 
# Stock Trading Strategy
######################################################################################################


# Resources Consulted:
# piazza course page (all)
# http://quantsoftware.gatech.edu/MC3-Project-1#Hints_.26_resources
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# https://www.youtube.com/watch?v=0fEcg_ZsYNY
# http://pandas.pydata.org/pandas-docs/stable/timeseries.html
# http://stackoverflow.com/questions/28356492/how-to-create-all-zero-dataframe-in-python
# http://pandas.pydata.org/pandas-docs/stable/indexing.html
# http://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
# https://piazza.com/class/idadrtx18nie1?cid=1383
# https://piazza.com/class/idadrtx18nie1?cid=1370
# https://piazza.com/class/idadrtx18nie1?cid=1332
# https://piazza.com/class/idadrtx18nie1?cid=1326
# https://piazza.com/class/idadrtx18nie1?cid=1366
# https://piazza.com/class/idadrtx18nie1?cid=1219



# import required libraries
import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

# disable SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
pd.options.mode.chained_assignment = None  



# PROJECT OVERVIEW:

# TODO: Impliment Technical Indicator 1 
# Normalized SMA

# TODO: Impliment Technical Indicator 2
# Normalized BB

# TODO: Impliment Technical Indicator 3
# Normalized Std

# TODO: Update/Add market simulator code

# TODO: Update/Add portfolio analysis code

# TODO: Modify plot functionality to support charts listed on wiki

# TODO: Train and Test learner / strategy for SINE data set

# TODO: Train and test learner / strategy for IBM data set

# TODO: use prev. sine ouput to backtest SINE strategy (trading_strategy_orders.csv)

# TODO: use prev. ibm ouput to backtest IBM strategy (trading_strategy_orders.csv)

# TODO: finish report, and submit to T-Square (8 page limit, 12 chart max, 10pt font)






######################################################################################################
# UTILITY FUNCTIONS

"""MLT: Utility code."""

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))



def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df



def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()






















######################################################################################################
# MC2-Project 1 Code (MARKET SIMULATOR)


def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # SETTING UP ORDERS DATAFRAME
    # Read orders file into a dataframe http://pandas.pydata.org/pandas-docs/stable/io.html#io-read-csv-table
    orders = pd.read_csv(orders_file)                       
    symbols = np.unique(orders['Symbol']).tolist()          # List of all the symbols used in orders

    # SETTING UP PRICES DATAFRAME
    # Read in adjusted closing prices for given symbols, date range... drop non-trading days... add cash column
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates, False).dropna()
    prices['cash'] = 1.00

    # SETTING UP TRADES DATAFRAME
    # Daily snapshot of portfolio changes (+ = Buy Order, - = Sell Order) with cash adjustments
    trades = pd.DataFrame(0.00, index=prices.index, columns=symbols)
    trades['cash'] = 0.00
    for row_index, row in orders.iterrows():
        try:
            if row.Order == 'SELL':
                trades.ix[row.Date,row.Symbol] += (-1 * row.Shares) # Subtract ShareAmount for Sell 
                trades.ix[row.Date,'cash'] += (row.Shares * prices.ix[row.Date, row.Symbol]) #adjust cash value for Sell
            elif row.Order == 'BUY':
                trades.ix[row.Date,row.Symbol] += (row.Shares) # Add ShareAmount for Buy
                trades.ix[row.Date,'cash'] += (-1 * row.Shares * prices.ix[row.Date, row.Symbol]) #adjust cash value for Buy
            else:
                print 'ERROR: order type not recognized, looking for BUY or SELL'
        except:
            print 'Unknown Error: date symbol mismatch, check that orders are within daterange'


    # SETTING UP HOLDINGS DATAFRAME 
    # accumulating trades into holdings dataframe, snapshot of shares and cash for given day
    holdings = pd.DataFrame(0.00, index=prices.index, columns=symbols)
    holdings['cash'] = 0.00
    holdings.ix[start_date,'cash'] = start_val      # add starting cash value
    previous_row = holdings.iloc[0]
    for row_index, row in holdings.iterrows():
        holdings.ix[row_index] = previous_row + trades.ix[row_index]    #previous day's value + trades
        previous_row = row

    #SETTING UP VALUES DATAFRAME
    # convert shares into their respective dollar amounts
    values = pd.np.multiply(holdings, prices)
    #DAILY VALUE OF THE PORTFOLIO
    portvals = values.sum(axis=1) 
    return portvals

















######################################################################################################
# MC2-Project 1 Code (PORTFOLIO ANALYSIS CODE)



def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices, allocations and starting value.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)

    Returns
    -------
        port_val: daily portfolio value
    """
    # Lecture 01-07
    # normalized data frame
    normed = prices/prices.ix[0,:]
    # normalized allocations data frame
    alloced = normed * allocs
    # position values data frame
    pos_vals = alloced * start_val
    # total value of the portfoli at a given day
    port_val = pos_vals.sum(axis=1)
    return port_val




def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    # Lecture 01-07
    # Portfolio Statistics
    daily_ret = (port_val / port_val.shift(1)) - 1
    cum_ret = (port_val[-1] / port_val[0]) - 1
    std_daily_ret = daily_ret.std()
    avg_daily_ret = daily_ret.mean()

    # annualized based on frequency of sampleing (daily)
    k = np.sqrt(samples_per_year)
    sharpe_ratio = k * np.mean(avg_daily_ret - daily_rf) / std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio




def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    # normalized data frame
    df = df/df.ix[0,:]
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.text
    dfp = df.plot(title=title, fontsize=14)
    # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xlabel
    dfp.set_xlabel(xlabel)
    dfp.set_ylabel(ylabel)
    plt.show()




def assess_portfolio(start_date, end_date, symbols, allocs, start_val=1):
    """Simulate and assess the performance of a stock portfolio."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = get_portfolio_value(prices, allocs, start_val)
    #plot_data(port_val, title="Daily Portfolio Value")

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocs
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", std_daily_ret
    print "Average Daily Return:", avg_daily_ret
    print "Cumulative Return:", cum_ret

    # Compare daily portfolio value with SPY using a normalized plot
    df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_temp, title="Daily Port Value vs. SPY")






















######################################################################################################
# MC3 - Project1 (Updated KNN Leaner for expanded stock data array)
class KNNLearner():

    #Constructor
    def __init__(self, k):
        self.k = k

    #Training Step
    def addEvidence(self, Xtrain, Ytrain):     
        data_model = np.ones([Xtrain.shape[0],Xtrain.shape[1]+1])
        data_model[:,0:Xtrain.shape[1]] = Xtrain
        data_model[:,Xtrain.shape[1]] = Ytrain
        self.data_model = data_model
        
    #Query
    def query(self, Xtest):
        k = self.k
        KNN = np.empty([k,self.data_model.shape[1]])
        Y = Xtest.sum(axis = 1)

        # iterate across X test data
        for point in range(0, Xtest.shape[0]):

            # iterate acorss data model created from training step
            for row in range(0, self.data_model.shape[0]):
                distance = np.linalg.norm( [Xtest[point] - self.data_model[row, 0:self.data_model.shape[1]-1]] )

                # get nearest neighbors
                for num in range(0, KNN.shape[0]):
                    if (row == num):
                        KNN[num, :] = self.data_model[row, :]
                        break

                    elif distance < ( np.linalg.norm([Xtest[point] - KNN[num, 0:KNN.shape[1]-1]]) ):
                        KNN[num+1:, :] = KNN[num: k-1, :]
                        KNN[num, :] = self.data_model[row,:]
                        break

            # Take the mean of the closest k points' Y values to make prediction
            Y[point] = np.mean(KNN[:, KNN.shape[1]-1])
            KNN = np.empty([k, self.data_model.shape[1]])

        return Y
























#################################################################################################################
# MC3 - Project 2 Code 


def trading_learner(symbol, training_start_date, training_end_date, test_start_date, test_end_date):

    # TRAINING STEP
    # Simulate a $SPX-only reference portfolio to get stats using training date range
    prices_data = get_data([symbol], pd.date_range(training_start_date, training_end_date))
    prices_data = prices_data[[symbol]]

    # Get Training Technical Indicators (SMA, Bollinger Bands, Std Dev)
    SMA = pd.DataFrame(index=prices_data.index, columns=['SMA','UB','LB'])  
    # X Set
    trainX = np.ones([(len(prices_data) - 25), 3])
    # Y Set
    trainY = np.ones([(len(prices_data) - 25)])

    counter = 19
    while (counter < len(prices_data)):
        # set up temp df for slicing
        temp_sma_df = prices_data.ix[counter - 19 : counter + 1][symbol]
        # GET SMA
        SMA.ix[counter]['SMA'] = temp_sma_df.mean()
        # GET Upper BBand
        SMA.ix[counter]['UB'] = temp_sma_df.mean() + (2 * temp_sma_df.std())
        # GET Lower BBand
        SMA.ix[counter]['LB'] = temp_sma_df.mean() - (2 * temp_sma_df.std())
        
        # Normalize Indicator Data 
        if (counter < len(prices_data) - 5):
            
            # X1 Indicator: Normalized Boilinger Band Feature
            # bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
            trainX[counter - 20, 0] = (prices_data.ix[counter][symbol] - temp_sma_df.mean()) / (2 * temp_sma_df.std())
           
            # X2 Indicator: Normalized Momentum
            # momentum[t] = (price[t]/price[t-N]) - 1
            trainX[counter - 20, 1] = ((prices_data.ix[counter][symbol] / prices_data.ix[counter - 19][symbol])) - 1

            # X3 Indicator: Rolling Standard Deviation of SMA 
            trainX[counter - 20, 2] = temp_sma_df.std()

            # Predicted Y Value (training set)
            trainY[counter - 20] = ((prices_data.ix[counter + 5][symbol] / prices_data.ix[counter][symbol])) - 1
        
        counter = counter + 1



    # TESTING STEP
    # Simulate a $SPX-only reference portfolio to get stats using testing date range
    prices_data = get_data([symbol], pd.date_range(test_start_date, test_end_date))
    prices_data = prices_data[[symbol]]  

    # Get Testing Technical Indicators (SMA, Bollinger Bands, Std Dev)
    SMA = pd.DataFrame(index=prices_data.index, columns=['SMA','UB','LB']) 
    # X Set
    testX = np.ones([len(prices_data) - 25, 3])
    # Y Set
    testY = np.ones([(len(prices_data) - 25)])


    counter = 19
    while (counter < len(prices_data)):
        temp_sma_df = prices_data.ix[counter - 19: counter + 1][symbol]
        # Get SMA
        SMA.ix[counter]['SMA'] = temp_sma_df.mean()

        # Normalize Indicator Data
        if (counter < len(prices_data) -5):

            # X1 Indicator: Normalized Boilinger Band Feature
            # bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
            testX[counter - 20, 0] = (prices_data.ix[counter][symbol] - temp_sma_df.mean()) / ( 2 * temp_sma_df.std())

            # X2 Indicator: Normalized Momentum
            # momentum[t] = (price[t]/price[t-N]) - 1
            testX[counter - 20, 1] = ((prices_data.ix[counter][symbol] / prices_data.ix[counter - 19][symbol])) - 1

            # X3 Indicator: Rolling Standard Deviation of SMA 
            testX[counter - 20, 2] = temp_sma_df.std()

            # Predicted Y Value (test set)
            testY[counter - 20] = (prices_data.ix[counter + 5][symbol]/prices_data.ix[counter][symbol])-1

        counter = counter + 1





    # KNN LEARNING STEP
    # Create / init KNN Learner
    learner = KNNLearner(k=3) 
    # Traing the learner
    learner.addEvidence(trainX, trainY) 
    # Get Predictions 
    predY = learner.query(testX) 
    # Get Stats for analysis
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    c = np.corrcoef(predY, y=testY)

    print '---------------------------'
    print 'Training'
    print "Data Range: {} to {}".format(training_start_date, training_end_date)
    print '---------------------------'
    print
    print "In sample results"
    print "RMSE: ", rmse
    print "corr: ", c[0,1]
    print
    print
    print







    # Plotting Step: Training Y, Price, and Predicted Y (3 colors)
    df_Y = pd.DataFrame(index = SMA.index,columns = ['Price','Predicted Y','Training Y'])

    counter = 19
    while (counter < len(prices_data)):
        df_Y.ix[counter]['Price'] = prices_data.ix[counter][symbol] 

        if (counter < len(prices_data) -5):
            df_Y.ix[counter]['Training Y'] = testY[counter-20]*prices_data.ix[counter][symbol] +prices_data.ix[counter][symbol] 
            df_Y.ix[counter]['Predicted Y'] = predY[counter-20]*prices_data.ix[counter][symbol] +prices_data.ix[counter][symbol] 
        counter = counter + 1
    Yplot = df_Y.plot(title=" Training-Y/ Price/ Predicted-Y Chart", fontsize = 14)
    Yplot.set_xlabel("Date")
    Yplot.set_ylabel("Price")
    Yplot.legend(loc = 'upper left')
    plt.show()





    # Write Trading Strategy To CSV File for BackTesting
    csvfile = open('trading_strategy_orders.csv', 'wb')
    orders_CSV = csv.writer(csvfile)
    orders_CSV.writerow(['Date','Symbol','Order','Shares'])


    # SECONDARY PLOTTING STEP: Entries and Exits CHART
    df_temp = pd.concat([prices_data[symbol]], axis=1)
    ax = df_temp.plot(title="Learner Entries and Exits", fontsize = 14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc = 'upper left')


    

    # Set up Helper Variables for trading strategy
    current_day = 0
    current_value = 0
    current_indicator = 0
    indicator = ""
    
    buying_day = 0
    estimated_gain = 0
    holding_period = 0
    
    count = 0
    shares = 0
    i = 0



    # KNN LEARNER BASED TRADING STRATEGY 
    # Holding period = 5 days
    # Predicted gain/loss threshold = .05%)

    orders = []
    while (i < predY.shape[0]):

        # IF Indicators suggest predicted price movement is greater than estimated gain, set holding period (ex: 4 days, 10 days...)
        if ((np.absolute(predY[i]*prices_data.ix[i][symbol]+(prices_data.ix[i][symbol]-prices_data.ix[buying_day][symbol])) > estimated_gain) and (indicator != "")):
            holding_period = 5
            estimated_gain = np.absolute(predY[i]*prices_data.ix[i][symbol]+(prices_data.ix[i][symbol]-prices_data.ix[buying_day][symbol]))

        # ESLE deincriment holding period counter
        else:
            holding_period = holding_period - 1



        # Set threshold for triggering buy/sell actions (ex: 1%, .75%, 2%...)
        if ((predY[i] < - .005 ) & (indicator == "")):
            plt.axvline(x = SMA[i+20:i+21].index[0],color = 'r') # Red = short entry
            indicator = "SELL"
            orders_CSV.writerow([SMA[i+20:i+21].index[0],symbol, indicator,100])
            count = count + 1
            holding_period = 5
            buying_day = i
            estimated_gain = -predY[i]*prices_data.ix[buying_day][symbol]



        # Set threshold for triggering buy/sell actions (ex: 1%, .75%, 2%...)
        elif((predY[i] > .005) & (indicator == "")):
            plt.axvline(x = SMA[i+20:i+21].index[0],color = 'g') # Green = long entry
            indicator = "BUY"
            orders_CSV.writerow([SMA[i+20:i+21].index[0],symbol, indicator,100])
            count = count + 1
            holding_period = 5
            buying_day = i + 20
            estimated_gain = predY[i]*prices_data.ix[buying_day][symbol]



        # LONG EXIT
        elif ((indicator == "BUY") & (holding_period==0)):
            plt.axvline(x = SMA[i+20:i+21].index[0],color = 'k') # Black = Exit
            indicator = ""
            orders_CSV.writerow([SMA[i+20:i+21].index[0],symbol,'SELL', 100])
            count = count + 1



        # SHORT EXIT
        elif ((indicator =="SELL") & (holding_period==0)):
            plt.axvline(x = SMA[i+20:i+21].index[0],color = 'k') # Black = Exit
            indicator = ""
            orders_CSV.writerow([SMA[i+20:i+21].index[0],symbol, 'BUY',100])
            count = count + 1

        i = i + 1

    plt.show()

































######################################################################################################
# MC3 - Project 2 Supporting / Backtesting Code (Modified from market simulator project)

def backtester(start_val, start_date, end_date):
    # get input data from CSV file
    orders_file = 'trading_strategy_orders.csv'

    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a SPY-only reference portfolio to get stats
    prices_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices_SPY = prices_SPY[['SPY']]  # remove SPY
    portvals_SPY = get_portfolio_value(prices_SPY, [1.0])
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(portvals_SPY)

    # Compare portfolio against SPY
    print '---------------------------'
    print 'Testing'
    print "Data Range: {} to {}".format(start_date, end_date)
    print '---------------------------'
    print
    print
    print "Out of Sample Results:"
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY: {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY: {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY: {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY: {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPY['SPY']], keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_temp, title="Daily Port Value vs. SPY")




























###################################################################################################################################

def test_run():
    """Driver function."""
    
    # STARTING PORTFOLIO VALUE ($10,000)
    start_val = 10000

    # TRAINING SET STARTING PARAMETERS (2008-2009)
    training_start_date = '2007-12-31'
    training_end_date = '2009-12-31'

    # FOR IN SAMPLE TESTING 
    #test_start_date = '2007-12-31'
    #test_end_date = '2009-12-31'

    # FOR OUT OF SAMPLE TESTING
    # TESTING SET STARTING PARAMETERS (2010)
    test_start_date = '2009-12-31'
    test_end_date = '2010-12-31'

    # SET OF SYMBOLS TO BE USED FOR TESTING/TRAINING/LEARNING
    #IBM DATA SET
    symbol = 'IBM'
    #SINE DATA SET
    #symbol = 'ML4T-399'
    
    # Learner/ Price Plots
    trading_learner(symbol, training_start_date, training_end_date, test_start_date, test_end_date)
    # Backtesting Plot
    backtester(start_val, test_start_date, test_end_date)


# # DRIVER FUNCTION: run 'python -m code' from command line in mc3_p2 directory
if __name__ == "__main__":
    test_run()

