import tweepy as tw
import pandas as pd
import numpy as np


def search(search_word, date_since, items, language, retweet=False):
    # Format de date: "YYYY-MM-DD"
    # items : le nombre de tweet max retourné

    auth = tw.OAuthHandler('API_Key', 'API_Secret_Key')
    auth.set_access_token('Access_Token',
                          'Access_Secret_Token')
    api = tw.API(auth)

    try:
        api.verify_credentials()
        print("Authentification ok")
    except tw.TweepError:
        print("Error during authentification")

    if not retweet:
        search_word = search_word + "-filter:retweets"

    tweets = tw.Cursor(api.search, q=search_word, lang=language, since=date_since, tweet_mode='extended').items(items)

    list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at, search_word] for tweet in tweets]

    tweet_data = pd.DataFrame(data=list_data, columns=["Text", "User", "Date", "Search Word"])

    return tweet_data