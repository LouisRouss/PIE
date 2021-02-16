import tweepy as tw
import pandas as pd
import numpy as np
import sys; sys.path.insert(1, '../functions')
import dict_utilities as dict_u

############################## Sub-functions ##############################

def get_df_news(data_folder, news_to_read, format_cols):
    '''Creates a dataframe from a ".parquet.gzip" file
    data_folder : directory of the parquet file
    news_to_read : "Bloomberg" or "Reuters"
    format_cols : temporary argument until we agree on the format of dataframe'''
    df = pd.read_parquet(data_folder + 'financial_data' + news_to_read + '.parquet.gzip')
    df = df.rename(columns = {'Article':'Text', 'Journalists':'Author'})
    return df[format_cols]

def search_twitter(search_word, date_since, nb_items, language, codes, format_cols, retweet=False):
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

def search_author(search_id,  date_since, nb_items, language, codes, format_cols, retweet=False):
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
