import TweetDataFrame as TDF
import Natural_Language_Processing as NLP
from sklearn.feature_extraction.text import CountVectorizer

Data = TDF.search("#Obama", "2020-11-13", 50, "en")
#print(Data)

Text=Data["Text"]
#Text.head()
#print(Text)
#sample=Text[0]
sample= "roses are red violets are up blue blue louis is a tortue "
sample=[sample]
#print(sample)

f= open("dict_mot/stop_words_english.txt","r")
stop_words_en=f.read().splitlines()
f.close

tokenizer=NLP.LemmaEnglishTokenizer(stop_words=stop_words_en,remove_non_words=False)

#' '.join(test)
countvect = CountVectorizer(tokenizer=tokenizer)
bow = countvect.fit_transform(sample)
#print(bow)
#print(countvect.get_feature_names())
#print(bow.toarray())

### print("Number of documents:", len(email_path))
words = countvect.get_feature_names()
#print("Number of words:", len(words))
#print("Document - words matrix:", bow.shape)
#print("First words:", words)

Data_fr = TDF.search("Obama", "2020-11-13", 50, "fr")
Text_fr=Data_fr["Text"]
#print(Data_fr)

sample_fr= Text_fr[0]
print(sample_fr)
tokenizer_fr=NLP.FrenchStemTokenizer(remove_non_words=False)
test=tokenizer_fr(sample_fr)
print(test)

