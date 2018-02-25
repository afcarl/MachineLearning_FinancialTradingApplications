"""MC1-P2: Optimize a portfolio."""
#Mark Trinquero
#CS 7646 - Project 1

import pandas as pd
import numpy as np
import scipy.optimize as spo
from util import get_data, plot_data
from analysis import get_portfolio_value, get_portfolio_stats


def sharpe_func(allocs, prices):
    "compute sharpe ratio given allocs and price data"
    port_val = get_portfolio_value(prices, allocs, 1)
    port_stats = get_portfolio_stats(port_val)

    sharpe_ratio = port_stats[3]
    return (-1 * sharpe_ratio)



def find_optimal_allocations(prices):
    """Find optimal allocations for a stock portfolio, optimizing for Sharpe ratio.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio

    Returns
    -------
        allocs: optimal allocations, as fractions that sum to 1.0
    """
    # TODO: Your code here
    #length of symbols to optimize
    noa = len(prices.columns)
    #initial guess on allocations (equal weight distribution)
    guess = noa * [1. / noa,]
    #constraint that all the allocations must sum to 1.0
    cons = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1})
    #bound that each allocation must be between 0.0 and 1.0
    bnds = tuple((0, 1) for x in range(noa))

    #call the optimizer to minimize the negative value of the sharpe ratio
    opts = spo.minimize(sharpe_func, guess, args=(prices,), method='SLSQP', options={'disp':True}, bounds=bnds, constraints=cons)
    allocs = opts['x']
    return allocs


def optimize_portfolio(start_date, end_date, symbols):
    """Simulate and optimize portfolio allocations."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get optimal allocations
    allocs = find_optimal_allocations(prices)
    allocs = allocs / np.sum(allocs)  # normalize allocations, if they don't sum to 1.0

    # Get daily portfolio value (already normalized since we use default start_val=1.0)
    port_val = get_portfolio_value(prices, allocs)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Optimal allocations:", allocs
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", std_daily_ret
    print "Average Daily Return:", avg_daily_ret
    print "Cumulative Return:", cum_ret

    # Compare daily portfolio value with normalized SPY
    normed_SPY = prices_SPY / prices_SPY.ix[0, :]
    df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
    plot_data(df_temp, title="Daily Portfolio Value and SPY")


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    
    #Example 1
    #symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    #Example 2
    #symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    #Example 3
    #symbols = ['YHOO', 'XOM', 'GLD', 'HNZ']
    #Example 4
    #symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ']
    #Submission
    symbols = ['GOOG', 'AAPL', 'GLD', 'HNZ']
    
    # Optimize portfolio
    optimize_portfolio(start_date, end_date, symbols)


if __name__ == "__main__":
    test_run()
