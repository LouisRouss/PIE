import yfinance
import datetime
import pandas as pd

def retrieve_data_from_yahoo_finance(tickers, from_date=None, to_date=None,interval="1d", proxy=None,threads=True):
    """
    Retrieve from Yahoo Finance.

    Args:
        tickers (list): tickers
        from_date (datetime.datetime): start date
        to_date (datetime.datetime): end date
        period (str): period
        interval (str): interval
        proxy (str): Proxy server
        threads (bool): multi thread fetching

    Returns:
        _data (pd.Dataframe) with following format:
        datetime, open, low, high, close, volume

    """
    kwargs = {"tickers": tickers,
              "start": from_date if from_date is not None else None,
              "end": to_date if to_date is not None else None,
              "interval": interval,
              "group_by": "ticker",
              "auto_adjust": True,
              "prepost": False,
              "threads": threads,
              "proxy": proxy,
              "progress": False
              }
    _data = yfinance.download(**kwargs)
    assert not _data.empty, "Data empty"
    # Weird fix because somehow we get from_date - Bday(1) to to_date
    return _data#.loc[_data.index >= from_date, ['Open', 'Low', 'High', 'Close', 'Volume']]




