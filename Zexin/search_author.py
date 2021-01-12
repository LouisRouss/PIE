import tweepy as tw
import pandas as pd

def search_author(search_id, items, retweet=False, reply = False):

    # items : le nombre de tweet max retourné
    # codes = ['API_Key', 'API_Secret_Key', 'Access_Token','Access_Secret_Token']
    
    auth = tw.OAuthHandler(codes[0], codes[1])
    auth.set_access_token(codes[2],codes[3])
    api = tw.API(auth)
    
    try:
        api.verify_credentials()
        print("Authentification ok")
    except tw.TweepError:
        print("Error during authentification")

    tweets = tw.Cursor(api.user_timeline, screen_name = search_id, count = items, include_rts = retweet,exclude_replies = not reply, tweet_mode='extended').items(items)

    list_data = [[tweet.full_text, tweet.user.screen_name, tweet.created_at] for tweet in tweets]

    tweet_data = pd.DataFrame(data=list_data, columns=["Text", "Author", "Date"])

    return tweet_data
