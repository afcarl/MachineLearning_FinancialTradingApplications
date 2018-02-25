# ML4T - MC2P2 - BOLINGER BAND STRATEGY 
# Mark Trinquero 

# URLs consulted
# https://piazza.com/class/idadrtx18nie1?cid=767
# http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.rolling_mean.html
# http://pandas.pydata.org/pandas-docs/stable/computation.html

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data
from util import get_data, plot_data, symbol_to_path

# Trade Data Frame Initilizers - stores strategy for backtesting and analysis
counter = 0
columns = ['Date', 'Symbol', 'Order', 'Shares']
trades_df = pd.DataFrame(columns= columns)

# Utility Functions
def get_rolling_mean(values, window):
    return pd.rolling_mean(values, window=window)

def get_rolling_std(values, window):
    return pd.rolling_std(values, window=window)

def get_bollinger_bands(rolling_mean, rolling_std):
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    return upper_band, lower_band


# MC2P1 Code - Market Simulator
def compute_portvals(start_date, end_date, trades_df, start_val):
    """Compute daily portfolio value given a sequence of orders in trades_df
    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # SETTING UP ORDERS DATAFRAME
    # Read orders file into a dataframe http://pandas.pydata.org/pandas-docs/stable/io.html#io-read-csv-table                    
    orders = trades_df
    symbols = np.unique(orders['Symbol']).tolist()          # List of all the symbols used in orders

    # SETTING UP PRICES DATAFRAME
    # Read in adjusted closing prices for given symbols, date range... drop non-trading days... add cash column
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates, addSPY=False).dropna()
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
            print 'Unknown Error:'


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


# MC2P2 Code - Bollinger Band Strategy Short & Long Indicators 

# https://piazza.com/class/idadrtx18nie1?cid=767
def get_indicator_data(metric, df):
    detector = np.pad(np.diff(np.array(df['IBM'] > metric).astype(int)), (1, 0), 'constant', constant_values=(0, ))
    return detector

def get_short_data(short_data):
    output = [i.tolist() for i in short_data][0]
    return output

def get_long_data(long_data):
    output = [i.tolist() for i in long_data][0][1:]
    return output


def get_shorts(upper_band, rolling_mean_IBM, df):
    global counter
    global trades_df
    entry_detector = get_indicator_data(upper_band, df)    
    exit_detector = get_indicator_data(rolling_mean_IBM, df)

    # Short Entries= crosses Upper Band from above
    entries = get_short_data(np.where(entry_detector == -1))    
    # Short Exits= crosses rolling_mean from above
    exits = get_short_data(np.where(exit_detector == -1))

    i, j = 0, 0  # entry index, exit index
    current_entry, current_exit = 0, 0
    while i < len(entries):
        while entries[i] < current_exit:
            i += 1
        plt.axvline(df.iloc[entries[i]].name, color='r') # RED bar for short entry
        trades_df.loc[counter] = [df.iloc[entries[i]].name, 'IBM', 'SELL', 100] # trigger ibm short sell per strategy
        counter += 1
        current_entry = entries[i]
        while j < len(exits):
            if exits[j] > current_entry:
                plt.axvline(df.iloc[exits[j]].name, color='k') # BLACK bar for short exit
                trades_df.loc[counter] = [df.iloc[exits[j]].name, 'IBM', 'BUY', 100] # trigger ibm buy per strategy
                counter += 1
                current_exit = exits[j]
                break
            j += 1
        i += 1


def get_longs(lower_band, rolling_mean_IBM, df):
    global counter
    global trades_df
    entry_detector = get_indicator_data(lower_band, df)
    exit_detector = get_indicator_data(rolling_mean_IBM, df)

    # Long Entries= crosses Lower Band from below
    entries = get_long_data(np.where(entry_detector == 1))
    # Long Exits = crosses rolling_mean from below
    exits = get_long_data(np.where(exit_detector == 1))

    i, j = 0, 0  # entry index, exit index
    current_entry, current_exit = 0, 0
    while i < len(entries):
        while entries[i] < current_exit:
            i += 1
        plt.axvline(df.iloc[entries[i]].name, color='g') # GREEN bar for long entry
        trades_df.loc[counter] = [df.iloc[entries[i]].name, 'IBM', 'BUY', 100] #trigger ibm buy per strategy
        counter += 1
        current_entry = entries[i]
        while j < len(exits):
            if exits[j] > current_entry:
                plt.axvline(df.iloc[exits[j]].name, color='k') # BLACK bar for long exit
                trades_df.loc[counter] = [df.iloc[exits[j]].name, 'IBM', 'SELL', 100] #trigger ibm sell per strategy
                counter += 1
                current_exit = exits[j]
                break
            j += 1
        i += 1


# # MC2P2 Test/ Driver Function (run 'python -m bollinger_strategy' from command line in mc2_p2 directory)
def test_run():
    """Driver function."""
    global trades_df
    # Define Date Parameters
    start_date = '2007-12-31'
    end_date = '2009-12-31'
    dates = pd.date_range(start_date, end_date)
    # Set Starting Cash Value
    start_val = 10000
    # Set Starting Symbols
    symbols = ['IBM']
    # Set up Prices DataFrame
    df = get_data(symbols, dates)
    # IBM rolling_mean (rolling mean)
    rolling_mean_IBM = get_rolling_mean(df['IBM'], window=20)
    # IBM rolling standard deviation
    rolling_std_IBM = get_rolling_std(df['IBM'], window=20)
    # Upper and Lower Bollinger Bands
    upper_band, lower_band = get_bollinger_bands(rolling_mean_IBM, rolling_std_IBM)
    # Plot raw SPY values, rolling mean and Bollinger Bands
    ax = df['IBM'].plot(title="Bollinger Bands", label='IBM')
    rolling_mean_IBM.plot(label='SMA', ax=ax, color='y')
    upper_band.plot(label='Upper Band', ax=ax, color='c')
    lower_band.plot(label='Lower Band', ax=ax, color='c')
    # Get short entries/exits
    get_shorts(upper_band, rolling_mean_IBM, df)
    # Get long entries/exits
    get_longs(lower_band, rolling_mean_IBM, df)
    
    # Process orders
    trades_df = trades_df.sort('Date')
    portvals = compute_portvals(start_date, end_date, trades_df, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot 1
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    # Plot 2
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")

if __name__ == "__main__":
    test_run()
