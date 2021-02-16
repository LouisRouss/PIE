import tweepy as tw
import pandas as pd
import numpy as np
import sys; sys.path.insert(1, '../functions')
import dict_utilities as dict_u
import re
import yfinance
import datetime
import pandas_market_calendars as mcal


############################## Sub-functions ##############################
def standardize_date(date, news_to_read):
    '''Returns a standard datetime64 format of given date taken from Reuters/Bloomberg dataset'''
    if (news_to_read == "Reuters") :
        date = re.split(' |, ', date)
        month2number = {"Jan" : '01', "Feb" : '02', "Mar" : '03', "Apr" : '04', "May" : '05', "Jun" : '06',
                        "Jul" : '07', "Aug" : '08', "Sep" : '09', "Oct" : '10', "Nov": '11', "Dec" : '12'}
        month = month2number[date[1]]
        day = date[2]
        year = date[3]
        hour_min = date[4]
        if 'p' in hour_min:
            hour_min = re.split(':|p', hour_min)
            hour, minutes = str((12+int(hour_min[0]))%12), hour_min[1]
        else:
            hour_min = re.split(':|a', hour_min)
            hour, minutes = hour_min[0], hour_min[1]

        hour = '0' + hour if len(hour)==1 else hour
        minutes = '0' + minutes if len(minutes)==1 else minutes
        day = '0' + day if len(day)==1 else day

        return np.datetime64(year+'-'+month+'-'+day+'T'+hour+':'+minutes)
    
    elif(news_to_read == "Bloomberg") :
        return np.datetime64(date[:-1])


def get_df_news(data_folder, news_to_read, format_cols = ["Text", "Author", "Date"]):
    '''Creates a dataframe from a ".parquet.gzip" file
    data_folder : directory of the parquet file
    news_to_read : "Bloomberg" or "Reuters"
    format_cols : temporary argument until we agree on the format of dataframe'''
    df = pd.read_parquet(data_folder + 'financial_data' + news_to_read + '.parquet.gzip')
    df = df.rename(columns = {'Article':'Text', 'Journalists':'Author'})
    df["Date"] = df["Date"].map(lambda date : standardize_date(date, news_to_read))
    return df[format_cols]

def search_twitter(search_word, date_since, nb_items, language, codes, format_cols= ["Text", "Author", "Date"], retweet=False):
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

    list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at, search_word] for tweet in tweets]

    tweet_df = pd.DataFrame(data=list_data, columns=["Text", "Author", "Date", "Search Word"])

    return tweet_df[format_cols]

def search_author(search_id,  date_since, nb_items, language, codes, format_cols = ["Text", "Author", "Date"], retweet=False):
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

    list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at] for tweet in tweets]

    tweet_data = pd.DataFrame(data=list_data, columns=["Text", "Author", "Date"])

    return tweet_data[format_cols]

def get_label(example, timeshift = 5, holidays = None):
    '''Returns the label of a line from a news dataset. 
    Time shifted 5h forward by default (US timebase)'''
    #We work with numpy datetime64, shifted to the US time
    date = np.datetime64(example["Date"]).astype("datetime64[m]")+np.timedelta64(timeshift, 'h')
    #Days to consider for data downloading : we need to be aware of non-business days
    next_day = np.busday_offset(date.astype("datetime64[D]"), 2, roll='forward', holidays = holidays)
    day_before = np.busday_offset(date.astype("datetime64[D]"), -2, roll='backward', holidays = holidays)

    old = date<(np.datetime64('today') - np.timedelta64(31, 'D'))
    #We get the data from Yahoo Finance (30m interval)
    if old :
        stock_data = retrieve_data_from_yahoo_finance("GOOGL", from_date=day_before.astype("datetime64[D]"),\
                                                          to_date=next_day.astype("datetime64[D]"), interval="1d")
    else :
        stock_data = retrieve_data_from_yahoo_finance("GOOGL", from_date=day_before.astype("datetime64[D]"),\
                                                          to_date=next_day.astype("datetime64[D]"), interval="30m")
    #We get the corresponding index of our data's date (using a generator is more time efficient)
    datetype = "datetime64[D]" if old else "datetime64[m]"
    tweet_idx = next(i for i in range(1, len(stock_data)-1) \
                     if date>=np.datetime64(stock_data.index.values[i]).astype(datetype) \
                     and date<np.datetime64(stock_data.index.values[i+1]).astype(datetype))

    if old :
        around_date = stock_data.iloc[tweet_idx-1:tweet_idx+2]
        #Estimation of second derivative
        label = around_date[2] - 2*around_date[1] + around_date[0]
    else:
        #We get the data +-4h (+-8 items since it's 30m slices) around our news
        around_date = stock_data.iloc[tweet_idx-8:tweet_idx+9]
        #We chose to compute the difference between areas after and before
        sum_before = np.sum(around_date[0:8])
        sum_after = np.sum(around_date[8:-1])
        label = sum_after-sum_before
    return label

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

def add_tweets_to_dict (date_since, nb_items, language, codes, dict_dir, label = True, format_cols = ["Text", "Author", "Date"], retweet=False, from_words=[], from_ids=[]):
    '''Adds tweets with matching search words across twitter and/or from specific accounts to the data dictionary with correct ticker
    date_since : date from which to search (format YYYY-MM-DD)
    nb_items : number of tweets to get
    language : "en", "fr"...
    codes : ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token'] (use get_codes)
    format_cols : temporary argument until we agree on the format of dataframe
    dict_dir : Directory of data dictionary
    from_words contains words to search all across twitter;
    from_ids contains the ids of Twitter accounts to search from'''
    
    #We take market holidays into account
    nyse = mcal.get_calendar('NYSE')
    holidays = nyse.holidays().holidays
    
    for search_word in from_words:
        df = search_twitter(search_word, date_since, nb_items, language, codes, format_cols, retweet=False)
        if label :
            df = df.assign(Label=df.apply(get_label, axis=1, args=(5, holidays)))
        dict_u.add_to_dict(df, search_word, dict_dir)
    for search_id in from_ids:
        df = search_author(search_id, date_since, nb_items, language, codes, format_cols, retweet=False)
        if label :
            df = df.assign(Label=df.apply(get_label, axis=1, args=(5, holidays)))
        dict_u.add_to_dict(df, search_id, dict_dir)
        
        
def add_news_to_dict(search_words, data_folder, news_to_read, dict_dir, label = True, format_cols = ["Text", "Author", "Date"]):
    '''Adds tweets with matching search words from parquet file
    data_folder : directory of the parquet file
    news_to_read : "Bloomberg" or "Reuters"
    dict_dir : Directory of data dictionary
    format_cols : temporary argument until we agree on the format of dataframe'''
    source_df = get_df_news(data_folder, news_to_read)
    #We take market holidays into account
    nyse = mcal.get_calendar('NYSE')
    holidays = nyse.holidays().holidays
    for search_word in search_words:
        filtered_df = source_df[source_df["Text"].apply(lambda article : search_word.lower() in article.lower())]
        if label:
            filtered_df = filtered_df.assign(Label=filtered_df.apply(get_label, axis=1, args=(0, holidays)))
        dict_u.add_to_dict(filtered_df, search_word, dict_dir)
 
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
    return _data["Close"]#.loc[_data.index >= from_date, ['Open', 'Low', 'High', 'Close', 'Volume']]