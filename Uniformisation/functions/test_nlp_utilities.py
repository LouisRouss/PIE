import nlp_utilities as nlpu
import pandas as pd

def test_get_tokenizer_fr_empty_stop_words():
    token = nlpu.get_tokenizer(language='fr')
    assert len(token.stopwords) > 0

def test_get_tokenizer_fr_empty_words():
    token = nlpu.get_tokenizer(language='fr')
    assert len(token.words) > 0
    
def test_tokenizer_fr_remove_stop_words():
#Test avec les stop words de base
    string = 'ils sont beaucoup à vivre pour le les à a pommes tiens ton'
    token = nlpu.get_tokenizer(language='fr')
    text = token(string)
    boo = True
    for word in token.stopwords:
        if word in text:
            boo = False
    assert boo
    
def test_tokenizer_fr_remove_non_words():
#Test avec les words de base
    string = 'ils blksjzh truc à vivre pour les pommes de terre'
    token = nlpu.get_tokenizer(language='fr')
    text = token(string)
    boo = True 
    for word in text:
        if word not in token.words:
            boo = False
    assert boo

def test_tokenizer_fr_punctuation():
    string = 'je , pense, que: la... vie'
    token = nlpu.get_tokenizer(language='fr')
    text = token(string)
    boo = True
    if ',' in text:
        boo = False
    elif 'pense,' in text:
        boo = False
    elif ':' in text:
        boo = False
    elif 'que:' in text:
        boo = False
    elif 'la...' in text:
        boo = False
    elif '...' in text:
        boo = False
    assert boo
    
def test_get_tokenizer_en_empty_stop_words():
    token = nlpu.get_tokenizer(language='en')
    assert len(token.stopwords) > 0
    
def test_get_tokenizer_en_empty_words():
    token = nlpu.get_tokenizer(language='en')
    assert len(token.words) > 0
    
def test_tokenizer_en_remove_stop_words():
#Test avec les stop words de base
    string = 'if they are those belongs to the main too American is the new bird of the amongs these themself'
    token = nlpu.get_tokenizer(language='en')
    text = token(string)
    boo = True
    for word in token.stopwords:
        if word in text:
            boo = False
    assert boo
    
def test_tokenizer_fr_remove_non_words():
#Test avec les words de base
    string = 'they are the main khjqsjkhjh of the USA but not only they kjhjh they also play golf'
    token = nlpu.get_tokenizer(language='en')
    text = token(string)
    boo = True 
    for word in text:
        if word not in token.words:
            boo = False
    assert boo

def test_tokenizer_en_punctuation():
    string = 'lame , Jhon, plane: car... life'
    token = nlpu.get_tokenizer(language='en')
    text = token(string)
    boo = True
    if ',' in text:
        boo = False
    elif 'Jhon,' in text:
        boo = False
    elif ':' in text:
        boo = False
    elif 'plane:' in text:
        boo = False
    elif 'car...' in text:
        boo = False
    elif '...' in text:
        boo = False
    assert boo
    
    
def test_df_to_bow_non_empty_bow():
    dataset = pd.read_csv('../dataset/dataset_sentiment.csv',sep=',',encoding='latin-1',header=None,names=['sentiment','Text'],nrows = 100)
    bow = nlpu.df_to_bow(dataset)
    assert not(bow.empty)
    

def test_df_to_bow_number_of_text():
    dataset = pd.read_csv('../dataset/dataset_sentiment.csv',sep=',',encoding='latin-1',header=None,names=['sentiment','Text'],nrows = 100)
    bow = bow = nlpu.df_to_bow(dataset)
    assert dataset.shape[0] == 100

    
    
    
    
    
    