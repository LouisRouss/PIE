import TweetDataFrame as TDF
import Natural_Language_Processing as NLP

Data = TDF.search("#Obama", "2020-11-13", 50, "en")
#print(Data)

Text=Data["Text"]
Text.head()
#print(Text)
#sample=Text[0]
sample= "roses are red violetts are up blue louis is a tortue "
print(sample)

f= open("dict_mot/stop_words_english.txt","r")
stop_words_en=f.read().splitlines()
f.close

tokenizer=NLP.LemmaEnglishTokenizer(stop_words=stop_words_en,remove_non_words=False)
test=tokenizer(sample)
print(test)

countvect = CountVectorizer(tokenizer=tokenizer)
bow = countvect.fit_transform(test)

### print("Number of documents:", len(email_path))
words = countvect.get_feature_names()
print("Number of words:", len(words))
print("Document - words matrix:", word_count.shape)
print("First words:", words[0:100])





