from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
f = open("stopwords.txt", "a")
for word in stop_words:
	f.write(word)
	f.write("\n")
f.close()
