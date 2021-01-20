import tweepy as tw
import pandas as pd
import numpy as np
import sys; sys.path.insert(1, '../functions')
from functions import dict_utilities as dict_u
import re
############################## Sub-functions ##############################

def get_df_news(data_folder, news_to_read, format_cols):
    '''Creates a dataframe from a ".parquet.gzip" file
    data_folder : directory of the parquet file
    news_to_read : "Bloomberg" or "Reuters"
    format_cols : temporary argument until we agree on the format of dataframe'''
    df = pd.read_parquet(data_folder + 'financial_data' + news_to_read + '.parquet.gzip')
    df = df.rename(columns = {'Article':'Text', 'Journalists':'Author'})
    return df[format_cols]

def search_twitter(search_word, date_since, nb_items, language, codes, format_cols,remove_URL = False, retweet=False):
    '''Constructs a dataframe of tweets found all over twitter with matching search word
    search_word : word to search in tweets
    date_since : date from which to search (format YYYY-MM-DD)
    nb_items : number of tweets to get
    language : "en", "fr"...
    codes : ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token'] (use get_codes)
    format_cols : temporary argument until we agree on the format of dataframe'''

    auth = tw.OAuthHandler(codes[0], codes[1])
    auth.set_access_token(codes[2],codes[3])
    api = tw.API(auth)

    try:
        api.verify_credentials()
    except tw.TweepError:
        print("Error during authentification")

    if not retweet:
        search_word = search_word + "-filter:retweets"

    tweets = tw.Cursor(api.search, q=search_word, lang=language, since=date_since, tweet_mode='extended').items(nb_items)
    if not remove_URL:
        list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at, search_word] for tweet in tweets]
    else:
        list_data = [[re.sub(r"http\S+", "",tweet.full_text), tweet.user.screen_name, tweet.created_at, search_word] for tweet in tweets]
        
    tweet_df = pd.DataFrame(data=list_data, columns=["Text", "Author", "Date", "Search Word"])

    return tweet_df[format_cols]

def search_author(search_id,  date_since, nb_items, language, codes, format_cols,remove_URL = False, retweet=False):
    '''Constructs a dataframe of the last ~nb_items tweets found since date_since on account with id search_id
    search_id : id of account to search from
    date_since : date from which to search (format YYYY-MM-DD)
    nb_items : number of tweets to get
    language : "en", "fr"...
    codes : ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token'] (use get_codes)
    format_cols : temporary argument until we agree on the format of dataframe'''
    
    auth = tw.OAuthHandler(codes[0], codes[1])
    auth.set_access_token(codes[2],codes[3])
    api = tw.API(auth)
    
    try:
        api.verify_credentials()
    except tw.TweepError:
        print("Error during authentification")

    tweets = tw.Cursor(api.user_timeline, screen_name = search_id, count = nb_items, include_rts = retweet, lang=language, since=date_since, tweet_mode='extended').items(nb_items)
    if not remove_URL:
        list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at] for tweet in tweets]
    else:
        list_data = [[re.sub(r"http\S+", "",tweet.full_text), tweet.user.screen_name, tweet.created_at] for tweet in tweets]

    tweet_data = pd.DataFrame(data=list_data, columns=["Text", "Author", "Date"])

    return tweet_data[format_cols]

############################## Add news to the data dictionary ##############################

def get_codes(codes_dir):
    '''Returns ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token'] used for Twitter API. They are stored in codes_dir txt file'''
    codes = []
    f = open(codes_dir, "r")
    for _ in range(4):
        string = f.readline()
        codes.append(string[string.find(':')+2:].strip())
    f.close()
    return codes

def add_tweets_to_dict (date_since, nb_items, language, codes, format_cols, dict_dir, retweet=False, from_words=[], from_ids=[]):
    '''Adds tweets with matching search words across twitter and/or from specific accounts to the data dictionary with correct ticker
    date_since : date from which to search (format YYYY-MM-DD)
    nb_items : number of tweets to get
    language : "en", "fr"...
    codes : ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token'] (use get_codes)
    format_cols : temporary argument until we agree on the format of dataframe
    dict_dir : Directory of data dictionary
    from_words contains words to search all across twitter;
    from_ids contains the ids of Twitter accounts to search from'''
    for search_word in from_words:
        df = search_twitter(search_word, date_since, nb_items, language, codes, format_cols, retweet=False)
        dict_u.add_to_dict(df, search_word, dict_dir, format_cols)
    for search_id in from_ids:
        df = search_author(search_id, date_since, nb_items, language, codes, format_cols, retweet=False)
        dict_u.add_to_dict(df, search_id, dict_dir, format_cols)
        
        
def add_news_to_dict(search_words, data_folder, news_to_read, dict_dir, format_cols):
    '''Adds tweets with matching search words from parquet file
    data_folder : directory of the parquet file
    news_to_read : "Bloomberg" or "Reuters"
    dict_dir : Directory of data dictionary
    format_cols : temporary argument until we agree on the format of dataframe'''
    source_df = get_df_news(data_folder, news_to_read, format_cols)
    for search_word in search_words:
        filtered_df = source_df[source_df[format_cols[0]].apply(lambda article : search_word.lower() in article.lower())]
        dict_u.add_to_dict(filtered_df, search_word, dict_dir, format_cols)
 
############################## Get stock market data ##############################

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

def create_up_down(ticker, lookback_minutes = 30,lookforward_minutes = 5,up_down_factor=2.0,percent_factor=0.01, start=None, end=None):
    """
    Creates a Pandas DataFrame that imports and calculates
    the percentage returns of an intraday Open Low High Close ticker in last 30 natural days.
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
    
def auto_label(ticker,tweet_id, codes_dir, format_cols):
    dt = search_author(search_id = tweet_id,  date_since = None, nb_items = 200, language = 'en', codes = get_codes(codes_dir), format_cols = format_cols, retweet=False)
    dm = create_up_down(ticker)
    dm.reset_index(inplace = True)
    dm['Date'] = dm.Datetime.dt.tz_convert('UTC')
    dm['Date'] = dm.Date.dt.tz_localize(None)
    dm.drop(columns = 'Datetime',inplace = True)
    data = pd.concat([dt,dm],names = 'Date').sort_values('Date')
    data['Hour'] = data.Date.dt.strftime('%H')
    df = data[(data['Date']>=(datetime.datetime.today()- datetime.timedelta(days = 30))) & (data['Hour']>='14') & (data['Hour']<='21')].copy()
    df.reset_index(drop = True, inplace = True)
    df['label'] = df['UpDown'].shift(-1)
    return df[df['UpDown'].isna()==True][['Text','Author','Date','label']].reset_index(drop = True)
