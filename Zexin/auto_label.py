import retrieve_data
import creat_up_down
import search_author
import pandas as pd
import datetime

def auto_label(ticker,tweet_id):
    dt = search_author(tweet_id,200)
    dm = creat_up_down(ticker)
    dm.reset_index(inplace = True)
    dm['Date'] = dm.Datetime.dt.tz_convert('UTC')
    dm['Date'] = dm.Date.dt.tz_localize(None)
    dm.drop(columns = 'Datetime',inplace = True)
    data = pd.concat([dt,dm],names = 'Date').sort_values('Date')
    data['Hour'] = data.Date.dt.strftime('%H')
    df = data[(data['Date']>=(datetime.datetime.today()- datetime.timedelta(days = 30)))&(data['Hour']>='14')&(data['Hour']<='21')].copy()
    df.reset_index(drop = True, inplace = True)
    df['label'] = df['UpDown'].shift(-1)
    return df[df['UpDown'].isna()==True][['Text','Author','Date','label']].reset_index(drop = True)
