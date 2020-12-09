# coding=utf-8

import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import rankdata


def make_partial_func(func, window=None, constant=None):
    ret_func = func
    if window:
        ret_func = partial(func, window=window)
        ret_func.__name__ = func.__name__ + "_" + str(window)
    if constant:
        ret_func = partial(func, constant=constant)
        ret_func.__name__ = func.__name__ + "_" + str(constant).replace('.', "__").replace("-", "___")
    return ret_func


def winsorize(x):
    """
    去极值
    """
    sigma = x.std()
    m = x.mean()
    x[x > m + 3 * sigma] = x + 3 * sigma
    x[x < m - 3 * sigma] = x - 3 * sigma
    return x


def standardize(x):
    """
    标准化
    """
    return (x - x.mean()) / x.std()


def if_then_else(condition, out1, out2):
    return out1 if condition else out2


# region Auxiliary functions
def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum()


def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()


def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()


def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def delta(df, window=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param window: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'window' days ago.
    """
    return df.diff(window)


def delay(df, window=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param window: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(window)


def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    # return df.rank(axis=1, pct=True)
    return df.rank(pct=True)


def scale(df, constant=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param constant: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = constant
    """
    return df.mul(constant).div(np.abs(df).sum())


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, window=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param window: the LWMA window
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:window, :] = df.iloc[:window, :]
    na_series = df.as_matrix()

    divisor = window * (window + 1) / 2
    y = (np.arange(window) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(window - 1, df.shape[0]):
        x = na_series[row - window + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])
