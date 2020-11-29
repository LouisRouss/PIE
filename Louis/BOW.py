import Natural_Language_Processing as NLP
from sklearn.feature_extraction.text import CountVectorizer

def BagOfWords(text,language, stop_words=None, remove_non_words=False, stemming=False):
    text=[text]
    if (stemming==True and language!= 'fr'):
        print("stemming only available in french")
        return(None)
     
    else:
        if (stemming==True):
               tokenizer=NLP.FrenchStemTokenizer(stop_words=stop_words,remove_non_words=remove_non_words)
        else:
               tokenizer=NLP.LemmaTokenizer(language, stop_words=stop_words,remove_non_words=remove_non_words)
    countvect = CountVectorizer(tokenizer=tokenizer)
    bow = countvect.fit_transform(sample_fr)
    words = countvect.get_feature_names()
    return((words,bow.toarray().reshape(-1)))