from sklearn.feature_extraction.text import CountVectorizer
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem.snowball import FrenchStemmer
import pandas as pd
from string import punctuation

# A voir ce qui est fait des mots avec un hashtag, peut être utiliser TweetTokenizer


class LemmaEnglishTokenizer(object):
    # An English lemmatization tokenizer
    def __init__(self,stop_words ,remove_non_words=True):
        self.wnl = WordNetLemmatizer()
        self.stopwords = stop_words
        self.words = set(words.words())
        self.remove_non_words = remove_non_words

    def __call__(self, doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        # remove non words
        if self.remove_non_words:
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word) > 1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.wnl.lemmatize(t) for t in word_list]


class LemmaTokenizer(object):
    # A multi language lemmatization tokenizer
    def __init__(self, language, stop_words=None, remove_non_words=True):
        self.wnl = WordNetLemmatizer()
        if language == 'fr':
            if stop_words is None:
                self.stopwords = set(stopwords.words('french'))
            else:
                self.stopwords = stop_words
        elif language == 'en':
            if stop_words is None:
                self.stopwords = set(stopwords.words('english'))
            else:
                self.stopwords = stop_words
        else:
            print('Wrong initialisation: language is not defined \n set as english')
            self.stopwords = set(stopwords.words('english'))
        self.words = set(words.words())
        self.remove_non_words = remove_non_words

    def __call__(self,doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        # remove non words
        if self.remove_non_words:
            word_list = [word for word in word_list if word in self.words]
        # remove 1 character words
        word_list = [word for word in word_list if len(word)>1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.wnl.lemmatize(t) for t in word_list]


class FrenchStemTokenizer(object):
    # A French Stemmer Tokenizer
    def __init__(self, remove_non_words=True):
        self.st = FrenchStemmer()
        self.stopwords = set(stopwords.words('french'))
        self.words = set(words.words())
        self.remove_non_words = remove_non_words

    def __call__(self, doc):
        # tokenize words and punctuation
        word_list = wordpunct_tokenize(doc)
        # remove stopwords
        word_list = [word for word in word_list if word not in self.stopwords]
        print(self.stopwords)
        
        # remove non words
        if (self.remove_non_words):
            word_list = [word for word in word_list if word in self.words]
        # remove 1-character words
        word_list = [word for word in word_list if len(word) > 1]
        # remove non alpha
        word_list = [word for word in word_list if word.isalpha()]
        return [self.st.stem(t) for t in word_list]
    
def BagOfWords(text,language, stop_words=None, remove_non_words=False, stemming=False):
    text=[text]
    if (stemming==True and language!= 'fr'):
        print("stemming only available in french")
        return(None)
     
    else:
        if (stemming==True):
               tokenizer=FrenchStemTokenizer(stop_words=stop_words,remove_non_words=remove_non_words)
        else:
               tokenizer=LemmaTokenizer(language, stop_words=stop_words,remove_non_words=remove_non_words)
    countvect = CountVectorizer(tokenizer=tokenizer)
    bow = countvect.fit_transform(text)
    words = countvect.get_feature_names()
    return((words,bow.toarray().reshape(-1)))    


def CreateData(words, bow):
    # Permet de créer un dataframe a partir d'un BOW
    new = pd.DataFrame(columns=words)
    new.loc[0] = bow
    return new


def AddInBow(data, words, bow):
    # Permet d'ajouter dans un dataframe une nouvelle ligne (nouveau bow) avec d'eventuels nouveaux mots(colonnes)
    new = CreateData(words, bow)
    data = pd.concat([data, new], sort=False)
    data = data.fillna('0')
    return data