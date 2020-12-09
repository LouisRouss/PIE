import datetime
import pandas as pd
import retrieve_data

def creat_up_down(ticker, lookback_minutes = 30,lookforward_minutes = 5,up_down_factor=2.0,percent_factor=0.01, start=None, end=None):
    """
    Creates a Pandas DataFrame that imports and calculates
    the percentage returns of an intraday OLHC ticker in last 30 natural days.
    ’lookback_minutes’ of prior returns are stored to create
    a feature vector, while ’lookforward_minutes’ are used to
    ascertain how far in the future to predict across.
    The actual prediction is to determine whether a ticker
    moves up by at least ’up_down_factor’ x ’percent_factor’,
    while not dropping below ’percent_factor’ in the same period.
    i.e. Does the stock move up 1% in a minute and not down by 0.5%?
    The DataFrame will consist of ’lookback_minutes’ columns for feature
    vectors and one column for whether the stock adheres to the "up/down"
    rule, which is 1 if True or 0 if False for each minute.

     Args:
        ticker (str): ticker
        lookback_minutes (int): lookback minutes
        lookforward_minutes (int): lookforward minutes
        up_down_factor (float): up down factor
        percent_factor (float): percent factor

    Returns:
        pd.DataFrame
    """

    today = datetime.datetime.today()
    data = pd.DataFrame()
    for i in range(int(30/7)+1):
        from_date = today - datetime.timedelta(days = 30 - 7*i)
        to_date = from_date + datetime.timedelta(days = 7)
        df = retrieve_data_from_yahoo_finance(tickers, from_date=from_date,to_date=to_date, interval="1m")
        data = pd.concat([data,df],axis = 0)
    path_file_data = tickers+"_donnees_"+today.strftime('%Y%m%d')+".csv"
#     if path_file_data:
#         data.to_csv(path_file_data)
    ts_df = data[['Close']].copy()
    ts_df["Lookback0"] = ts_df["Close"].pct_change() * 100
    for i in range(0, lookback_minutes):
        ts_df[f"Lookback{i + 1}"] = ts_df["Close"].shift(i + 1).pct_change() * 100
    for i in range(0, lookforward_minutes):
        ts_df[f"Lookforward{i + 1}"] = ts_df["Close"].shift(-(i + 1)).pct_change() * 100
    ts_df = ts_df.dropna()
    up_factor = up_down_factor * percent_factor
    down = percent_factor

    down_cols = [ts_df[f"Lookforward{i + 1}"] > -down for i in range(0, lookforward_minutes)]
    up_cols = [ts_df[f"Lookforward{i + 1}"] > up_factor for i in range(0, lookforward_minutes)]

    down_tot = down_cols[0]
    for col in down_cols[1:]:
        down_tot = down_tot & col
    up_tot = up_cols[0]
    for col in up_cols[1:]:
        up_tot = up_tot | col
    ts_df["UpDown"] = down_tot & up_tot
    ts_df["UpDown"] = ts_df["UpDown"].astype(int)
    path_file_updown = tickers+"_updown_"+today.strftime('%Y%m%d')+".csv"
#     if path_file_updown:
#         data.to_csv(path_file_updown)
    return ts_df
