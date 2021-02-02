import pandas as pd
import numpy as np
import pickle
import os
from nltk import wordpunct_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models import Word2Vec

class LemmaTokenizer(object):
    '''A tokenizer for english text'''
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
    '''A tokenizer for french text'''
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
    
    
def print_topics(model, feature_names, n_words):
    '''Print topics of given LDA or NMF model'''
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_words - 1:-1]])
        print(message)


def get_stop_words(words_to_add=[], words_to_delete=[], language = 'en'):
    '''Returns stop_words argument for get_tokenizer : uses default stop_words of given language to add words_to_add and delete words_to_delete '''
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

def get_tokenizer(stop_words = None, language = 'en'):
    '''Returns tokenizer of given langauge with given stop_words'''
    if language == 'fr':
        tokenizer = FrenchStemTokenizer(stop_words = stop_words)
    elif language == 'en':
        tokenizer = LemmaTokenizer(stop_words = stop_words)
    else :
        raise LanguageError("Language not supported")
    return tokenizer
        
def df_to_bow(df, stop_words = None, language = 'en', TFIDF = True):
    '''Returns the BOW (model and feature mapping) of a given "ticker" dataframe. 
    If TFIDF is true, the BOW is weighted with "Term frequency times inverse document frequency" 
    (generally better performances)'''
    text_df = df["Text"]
    tokenizer = get_tokenizer(stop_words, language)
    countvect = CountVectorizer(tokenizer = tokenizer, max_df=0.95, min_df=2)
    bow = countvect.fit_transform(text_df)
    feat2word = {v: k for k, v in countvect.vocabulary_.items()}
    if TFIDF :
        bow = TfidfTransformer().fit_transform(bow)
    columns = [feat2word[i] for i in range(len(feat2word))]
    bow = pd.DataFrame(bow.toarray())
    bow.columns = columns
    return bow

def df_to_bow_prediction(columns, df, stop_words = None ,language = 'en', TFIDF = True):
    '''Return the BOW of a DataFrame of texts , the BOW is made as such as a ML model trained on a training BOW can predict the sentiment of       these texts
      columns : the list of the words used in the training bow
      df : DataFrame of texts
      If TFIDF is true, the BOW is weighted with "Term frequency times inverse document frequency" 
      (generally better performances)
    '''
    dataframe_to_return = pd.DataFrame(columns=columns)
    bow_validation = df_to_bow(df,stop_words,language,TFIDF)
    columns_bow_validation = bow_validation.columns
    for word in columns_bow_validation:
        if word in columns:
            for i in range(bow_validation.shape[0]):
                dataframe_to_return.loc[i,word] = bow_validation[word][i]
    dataframe_to_return = dataframe_to_return.fillna(0)
    return dataframe_to_return
    
def df_to_vec(df, stop_words = None, language = 'en', size=200, window=5, min_count=1):
    '''Returns the Word2Vec model of a given "ticker" dataframe.'''
    text_df = df["Text"]
    tokenizer = get_tokenizer(text_df, stop_words, language)
    text_for_word2vec=[tokenizer(text) for text in text_df]
    model = Word2Vec(text_for_word2vec,size=size,window=window,min_count=min_count)
    return model

def df_to_lda(df, n_topics = 5, stop_words = None, language = 'en', TF = True):
    '''Returns the LDA model of given "ticker" dataframe'''
    text_df = df["Text"]
    tokenizer = get_tokenizer(stop_words, language)
    countvect = CountVectorizer(tokenizer = tokenizer, max_df=0.95, min_df=2)
    bow = countvect.fit_transform(text_df)
    feat2word = {v: k for k, v in countvect.vocabulary_.items()}
    if (TF):
        bow = TfidfTransformer(use_idf = False).fit_transform(bow)
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online',
                                    learning_offset=50., random_state=0)
    lda.fit(bow)
    return lda, countvect, feat2word

def df_to_nmf(df, n_topics=5, stop_words = None, language = 'en'):
    '''Returns the NMF model of given "ticker" dataframe'''
    bow, countvect, feat2word = df_to_bow(df, stop_words = stop_words, language = language, TFIDF = True)
    nmf = NMF(n_components=n_topics, alpha=.1, l1_ratio=.5).fit(bow)
    return nmf, countvect, feat2word