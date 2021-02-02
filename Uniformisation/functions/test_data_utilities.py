import data_utilities as data_u
import datetime

def test_search_twitter_number_of_tweet():
    codes = data_u.get_codes('../../../codes.txt')
    now = datetime.datetime.now()
    date = now.strftime('%Y/%m/%d')
    df = data_u.search_twitter('plane', date, 10, 'en', codes, ["Text", "Author", "Date", "Search Word"],remove_URL = True, retweet=False)
    assert df.shape[0] == 10
    
def test_search_twitter_date_since():
    codes = data_u.get_codes('../../../codes.txt')
    now = datetime.datetime.now()
    date = now.strftime('%Y/%m/%d')
    df = data_u.search_twitter('epiphany', date, 1000, 'en', codes, ["Text", "Author", "Date", "Search Word"], remove_URL = 
                               True,retweet=False) 
    test = True
    for i in range(df.shape[0]):
        if date not in str(df.Date.iloc[i]):
            test = False
    assert test
    
def test_search_twitter_keyword():
        #Twitter va chercher les synonymes du keyword aussi d'ou le fail du test si on prends un nom commun + twitter va chercher dans 
        #l'eventuelle URL linked
        codes = data_u.get_codes('../../../codes.txt')
        key = 'Obama'
        df = data_u.search_twitter(key,'2021/02/01', 10, 'en', codes, ["Text", "Author", "Date", "Search Word"],remove_URL = True, 
                                   retweet=False)
        test = True
        for i in range(df.shape[0]):
            texte = df.Text.iloc[i]
            if key.upper() not in texte.upper():
                test = False
        assert test
        
def test_search_author_number_of_tweet():
    codes = data_u.get_codes('../../../codes.txt')
    now = datetime.datetime.now()
    date = now.strftime('%Y/%m/%d')
    df = data_u.search_author('CNNBusiness', date, 10, 'en', codes, ["Text", "Author", "Date"],remove_URL = True, retweet=False)
    assert df.shape[0] == 10

def test_search_author_date_since():
    codes = data_u.get_codes('../../../codes.txt')
    now = datetime.datetime.now()
    date = now.strftime('%Y/%m/%d')
    df = data_u.search_author('CNNBusiness', date, 1000, 'en', codes, ["Text", "Author", "Date"], remove_URL = 
                               True,retweet=False)  
    test = True
    for i in range(df.shape[0]):
        if date not in str(df.Date.iloc[i]):
            test = False
    assert test

def test_search_author_right_author():
    codes = data_u.get_codes('../../../codes.txt')
    now = datetime.datetime.now()
    date = now.strftime('%Y/%m/%d')
    df = data_u.search_author('CNNBusiness', date, 10, 'en', codes, ["Text", "Author", "Date"], remove_URL = 
                               True,retweet=False)
    test = True
    for i in range(df.shape[0]):
        if df.Author.iloc[i] != 'CNNBusiness':
            test = False
    assert test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    