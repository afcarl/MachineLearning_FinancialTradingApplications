#Mark Trinquero
#python market simulator

#URLs consulted during development (per GA Tech honor code)
#http://pandas.pydata.org/pandas-docs/stable/timeseries.html
#http://stackoverflow.com/questions/28356492/how-to-create-all-zero-dataframe-in-python
#http://pandas.pydata.org/pandas-docs/stable/indexing.html
#http://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html

import pandas as pd
import numpy as np
import os

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

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
    print '-------------'
    print ' holdings'
    print '--------------'
    print holdings
    print '-------------'
    print ' prices'
    print '--------------'
    print prices
    print '-----------------'
    print ' orders'
    print orders
    print '-----------'

    values = pd.np.multiply(holdings, prices)
    #DAILY VALUE OF THE PORTFOLIO
    portvals = values.sum(axis=1) 
    return portvals




def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-14'
    end_date = '2011-12-14'
    orders_file = os.path.join("orders", "orders2.csv")
    start_val = 1000000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
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

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
