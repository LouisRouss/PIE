import retrieve_data
import creat_up_down
import search_author
import pandas as pd
import datetime

def auto_label(ticker, date,items, codes, format_cols,tweet_id = None, tweet_keyword = None,period = 3, language = 'en',retweet=False,reply = False):
    #ticker: stock ticker
    #date: searh date Problem:!! it seems that 'since' has been removed from tweetpy
    # items : le nombre de tweet max retourné
    # codes = ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token']
    # format_cols: list like ["Text", "Author", "Date"]
    # tweet_id: tweeter id of author we search
    # tweet_keyword: keyword of research
    # period : how many days of data
    # language : "en", "fr"...
    
    if tweet_id:
        dt = search_author(tweet_id,items,codes,format_cols,reply)
    else:
        dt = pd.DataFrame(columns = format_cols)
    if tweet_keyword:
        td = search_twitter(tweet_keyword,date,items,language,codes,format_cols,retweet)
    else:
        td = pd.DataFrame(columns = format_cols)

    dm = creat_up_down(ticker,date,period)
    dm.reset_index(inplace = True)
    dm['Date'] = dm.Datetime.dt.tz_convert('UTC')
    dm['Date'] = dm.Date.dt.tz_localize(None)
    dm.drop(columns = 'Datetime',inplace = True)
    dn = pd.concat([dt,td]).sort_values('Date')
    data = pd.concat([dn,dm]).sort_values('Date')
    data['Hour'] = data.Date.dt.strftime('%H')
    df = data[(data['Date']>=(pd.to_datetime(date)))&(data['Date']<=(pd.to_datetime(date) + datetime.timedelta(days = period)))&(data['Hour']>='14')&(data['Hour']<='21')].copy()
    df.reset_index(drop = True, inplace = True)
    df['label'] = df['UpDown'].shift(-1)
    return df[df['UpDown'].isna()==True][['Text','Author','Date','label']].reset_index(drop = True)
