"""MC1-P1: Analyze a portfolio."""
#Mark Trinquero
#CS 7646 - Project 1

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from util import get_data, plot_data


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
    dfp = df.plot(title=title, fontsize=8)
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
    plot_normalized_data(df_temp, title="Daily portfolio value and SPY")


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2010-01-01'
    end_date = '2010-12-31'

    symbol_allocations = OrderedDict([('GOOG', 0.2), ('AAPL', 0.2), ('GLD', 0.4), ('XOM', 0.2)])  # symbols and corresponding allocations
    #symbol_allocations = OrderedDict([('AXP', 0.0), ('HPQ', 0.0), ('IBM', 0.0), ('HNZ', 1.0)])  # allocations from wiki example

    symbols = symbol_allocations.keys()  # list of symbols, e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocs = symbol_allocations.values()  # list of allocations, e.g.: [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  # starting value of portfolio

    # Assess the portfolio
    assess_portfolio(start_date, end_date, symbols, allocs, start_val)


if __name__ == "__main__":
    test_run()
