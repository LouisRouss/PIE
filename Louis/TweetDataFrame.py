import tweepy as tw
import pandas as pd
import numpy as np
import re

def search(search_word, date_since, items, language,remove_URL = False, retweet=False):
    # Format de date: "YYYY-MM-DD"
    # items : le nombre de tweet max retourné

    auth = tw.OAuthHandler('...', '...')
    auth.set_access_token('...','...')
    api = tw.API(auth)

    try:
        api.verify_credentials()
        print("Authentification ok")
    except tw.TweepError:
        print("Error during authentification")

    if not retweet:
        search_word = search_word + "-filter:retweets"

    tweets = tw.Cursor(api.search, q=search_word, lang=language, since=date_since, tweet_mode='extended').items(items)
    if not remove_URL:
        list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at, search_word] for tweet in tweets]
    else:
        list_data = [[re.sub(r"http\S+", "",tweet.full_text), tweet.user.screen_name, tweet.created_at, search_word] for tweet in tweets]
    
    tweet_data = pd.DataFrame(data=list_data, columns=["Text", "User", "Date", "Search Word"])

    return tweet_data


def search_author(search_id, items,remove_URL = False, retweet=False, reply=False):
    # Format de date: "YYYY-MM-DD"
    # items : le nombre de tweet max retourné
    # codes = ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token']

    auth = tw.OAuthHandler('I5exKno5D4drEQK2uShWDRFfm', 'XNe8j4iWBPKDpl1jSgqawgCyxzscDvfneDMYibimLeu1ukwoGk')
    auth.set_access_token('1326944909207728129-S6ogCIOnY0fdTKJe4GyQm866rTsm0c', 'Np5teHlej9VsYrBueZtWmg9kZbxMKzEp8xtjlSJFT79Qy')
    api = tw.API(auth)

    try:
        api.verify_credentials()
        print("Authentification ok")
    except tw.TweepError:
        print("Error during authentification")

    tweets = tw.Cursor(api.user_timeline, screen_name=search_id, count=items, include_rts=retweet,
                       exclude_replies=not reply, tweet_mode='extended').items(items)
    if not remove_URL:
        list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at] for tweet in tweets]
    else:
        list_data = [[re.sub(r"http\S+", "",tweet.full_text), tweet.user.screen_name, tweet.created_at] for tweet in tweets]
    
    tweet_data = pd.DataFrame(data=list_data, columns=["Text", "Author", "Date"])

    return tweet_data
