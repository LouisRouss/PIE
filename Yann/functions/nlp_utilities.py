import pandas as pd
import numpy as np
import pickle
import os
from nltk import wordpunct_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import Word2Vec

class LemmaTokenizer(object):
    def __init__(self, stop_words = None, remove_non_words=True):
        self.wnl = WordNetLemmatizer()
        if stop_words is None:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = stop_words
        self.words = set(words.words())
        self.remove_non_words = remove_non_words
    def __call__(self, doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        # remove non words
        if(self.remove_non_words):
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word)>1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.wnl.lemmatize(t) for t in word_list]
    
class FrenchStemTokenizer(object):
    # A French Stemmer Tokenizer
    def __init__(self, stop_words=None, remove_non_words=True):
        self.st = FrenchStemmer()
        if stop_words is None:
            self.stopwords = set(stopwords.words('french'))
        else:
            self.stopwords = stop_words
        self.words = set(words.words())
        self.remove_non_words = remove_non_words

    def __call__(self, doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        
        # remove non words
        if (self.remove_non_words):
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word) > 1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.st.stem(t) for t in word_list]
    
def get_stop_words(words_to_add=[], words_to_delete=[], language = 'en'):
    if language == 'fr':
        stop_words = set(stopwords.words('french'))
    elif language == 'en':
        stop_words = set(stopwords.words('english'))
    else :
        raise LanguageError("Language not supported")

    for word_to_add in words_to_add :
        stop_words.append(word_to_add)
       
    for word_to_delete in words_to_delete:
        if word_to_delete in stop_words:
            stop_words.remove(word_to_delete)
        else:
            print(f'Word {word_to_delete} not in original stop words list\n')
    return stop_words
        
def df_to_bow(df, stop_words = None, language = 'en', TFIDF = True):
    '''Returns the BOW (model and feature mapping) of a given "ticker" dataframe. 
    If TFIDF is true, the BOW is weighted with "Term frequency times inverse document frequency" 
    (generally better performances)'''
    text_df = df["Text"]
    if language == 'fr':
        tokenizer = FrenchStemTokenizer(stop_words = stop_words)
    elif language == 'en':
        tokenizer = LemmaTokenizer(stop_words = stop_words)
    else :
        raise LanguageError("Language not supported")
    countvect = CountVectorizer(tokenizer)
    bow = countvect.fit_transform(text_df)
    feat2word = {v: k for k, v in countvect.vocabulary_.items()}
    if TFIDF :
        bow = TfidfTransformer().fit_transform(bow)
    return bow, countvect, feat2word

def df_to_vec(df, stop_words = None, language = 'en', size=200, window=5, min_count=1):
    '''Returns the Word2Vec model of a given "ticker" dataframe.'''
    text_df = df["Text"]
    if language == 'fr':
        tokenizer = FrenchStemTokenizer(stop_words = stop_words)
    elif language == 'en':
        tokenizer = LemmaTokenizer(stop_words = stop_words)
    else :
        raise LanguageError("Language not supported")
    text_for_word2vec=[tokenizer(text) for text in text_df]
    model = Word2Vec(text_for_word2vec,size=size,window=window,min_count=min_count)
    return model