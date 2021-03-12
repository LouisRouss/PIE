import pandas as pd
import numpy as np
import pickle
import os
import data_utilities as data_u
import dict_utilities as dict_u

from nltk import wordpunct_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models import Word2Vec
from joblib import dump, load

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score


######Pre-processing ########
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
        
########## Representation models ###########

models = ["TFIDF",
           "Word2Vec",
           "LDA",
           "NMF"]

def df_to_bow(df, stop_words = None, language = 'en', TFIDF = True):
    '''Returns the BOW (model and feature mapping) of a given "ticker" dataframe. 
    If TFIDF is true, the BOW is weighted with "Term frequency times inverse document frequency" 
    (generally better performances)'''
    text_df = df["Text"]
    tokenizer = get_tokenizer(stop_words, language)
    countvect = CountVectorizer(tokenizer = tokenizer, max_df=0.95, min_df=2)
    X = countvect.fit_transform(text_df)
    feat2word = {v: k for k, v in countvect.vocabulary_.items()}
    if TFIDF :
        X = TfidfTransformer().fit_transform(X)
    return X, countvect, feat2word

def get_bow_features(df, stop_words = None, language = 'en', TFIDF = True):
    return df_to_bow(df, stop_words, language, TFIDF)[0]

def df_to_vec(df, stop_words = None, language = 'en', size=200, window=5, min_count=1):
    '''Returns the Word2Vec model of a given "ticker" dataframe.'''
    text_df = df["Text"]
    tokenizer = get_tokenizer(stop_words, language)
    text_for_word2vec=[tokenizer(text) for text in text_df]
    model = Word2Vec(text_for_word2vec,size=size,window=window,min_count=min_count)
    
    def get_vect(word, model):
        try:
            return model.wv[word]
        except KeyError:
            return numpy.zeros((model.vector_size,))

    def word2vec_features(articles, model):
        features = np.vstack([sum(get_vect(word, model) for word in article) for article in articles])
        return features
    print(text_for_word2vec.shape)
    X = word2vec_features(text_for_word2vec, model)
    
    return X, model

def get_w2v_features(df, stop_words = None, language = 'en', size=200, window=5, min_count=1):
    return df_to_vec(df, stop_words, language, size, window, min_count)[0]

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
    X = lda.fit_transform(bow)
    return X, lda, countvect, feat2word

def get_lda_features(df, n_topics = 5, stop_words = None, language = 'en', TF = True):
    return df_to_lda(df, n_topics, stop_words, language, TF)[0]

def df_to_nmf(df, n_topics=5, stop_words = None, language = 'en'):
    '''Returns the NMF model of given "ticker" dataframe'''
    bow, countvect, feat2word = df_to_bow(df, stop_words = stop_words, language = language, TFIDF = True)
    nmf = NMF(n_components=n_topics, alpha=.1, l1_ratio=.5)
    X = nmf.fit_transform(bow)
    return X, nmf, countvect, feat2word

def get_nmf_features(df, n_topics=5, stop_words = None, language = 'en'):
    return df_to_nmf(df, n_topics, stop_words, language)[0]

model_name_to_function = { "TFIDF" : df_to_bow,
           "Word2Vec" : df_to_vec,
           "LDA" : df_to_lda,
           "NMF" : df_to_nmf}

def df_to_model(df, model_name):
    '''Returns default representation model of given dataframe. Available models are "TFIDF", "Word2Vec", "LDA", "NMF"'''
    if model_name in models:
        return model_name_to_function[model_name](df)
    else:
        print("Wrong model name")
        return 0
    
def get_model_features(df, model_name):
    '''Get features of default representation model of given dataframe. Available models are "TFIDF", "Word2Vec", "LDA", "NMF"'''
    if model_name in models:
        return df_to_model(df, model_name)[0]
    else:
        print("Wrong model name")
        return 0
        

######## Training ########

clfs = { "Random Forests" : RandomForestRegressor(n_estimators=100, criterion="mse"),
         "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100),
         "Decision Tree" : tree.DecisionTreeRegressor(),
         "SVR" : svm.SVR(kernel='rbf', C = 1),
         "Gaussian Process" : GaussianProcessRegressor(n_restarts_optimizer = 3),
         "Adaboost" : AdaBoostRegressor(tree.DecisionTreeRegressor(criterion='mse', max_depth = 3), n_estimators=100)}

data_folder = '../data/'

def save_model(clf,clf_name):
    '''Save the sklearn model at the path indicated, the path got to end with .joblib (the type of the saved file)'''
    path = f'../data/clfs/{clf_name}.joblib'
    dump(clf,path)

def load_model(clf_name):
    '''Load and return the sklearn model located at the path specified'''
    path = f'../data/models/{clf_name}.joblib'
    clf = load(path)
    return clf

def init_clf(clf_name):
    "Loads previously trained (or fresh) classifier/regressor"
    clf_path = f'../data/models/{clf_name}.joblib'
    if os.path.isfile(clf_path):
        clf = load(clf_path)
    else :
        clf = clfs[clf_name]
    return clf

def generate_input(model_name, dict_dir, date_since="2021-02-13", nb_items = 1000, language = 'en', from_ids=["Google"]):
    codes = data_u.get_codes(data_folder + "twitter_codes.txt")
    #Get news from twitter and add them to the data dictionary
    data_u.add_tweets_to_dict(date_since, nb_items, language, codes,\
                        dict_dir, retweet=False, from_ids=from_ids)
    #Get the data dictionary
    data_dict = dict_u.get_dict(dict_dir)
    #Get one of the dataframes from the data dictionary
    df = data_dict[from_ids[0].lower()]
    #Generate input with given representation model
    X = get_model_features(df, model_name)
    return X
    
def train(model_name, clf_name, dict_dir, date_since="2021-02-13", nb_items = 1000, language = 'en', from_ids=["Google"]):
    '''model_name is the NLP representation model among "TFIDF", "LDA", "NMF", "Word2Vec"
       clf_name is the classifier/regressor model among (for now) "Random Forests", "Gradient Boosting", 
       "Decision Tree", "SVR", "Gaussian Process", "Adaboost"'''
    
    clf = init_clf(clf_name)
    X = generate_input(model_name, dict_dir, date_since, nb_items, language, from_ids)
    clf.fit(X)
    save_model(clf,clf_name)
    print(f"Training done on {nb_items} news")
    

def predict(model_name, clf_name, dict_dir, date_since="2021-02-13", nb_items = 1000, language = 'en', from_ids=["Google"]):
    clf = init_clf(clf_name)
    X = generate_input(model_name, dict_dir, date_since, nb_items, language, from_ids)
    y_pred = clf.predict(X)
    return y_pred
    
