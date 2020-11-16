import tweepy as tw
import pandas as pd
import numpy as np

def get_codes(codes_dir):
    codes = []
    f = open(codes_dir, "r")
    for _ in range(4):
        string = f.readline()
        codes.append(string[string.find(':')+2:].strip())
    f.close()
    return codes

def search(search_word, date_since, nb_items, language, codes, retweet=False):
    # Format de date: "YYYY-MM-DD"
    # nb_items : le nombre de tweet max retourné
    # codes = ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token']

    auth = tw.OAuthHandler(codes[0], codes[1])
    auth.set_access_token(codes[2],codes[3])
    api = tw.API(auth)

    try:
        api.verify_credentials()
        print("Authentification ok")
    except tw.TweepError:
        print("Error during authentification")

    if not retweet:
        search_word = search_word + "-filter:retweets"

    tweets = tw.Cursor(api.search, q=search_word, lang=language, since=date_since, tweet_mode='extended').items(nb_items)

    list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at, search_word] for tweet in tweets]

    tweet_data = pd.DataFrame(data=list_data, columns=["Text", "User", "Date", "Search Word"])

    return tweet_data