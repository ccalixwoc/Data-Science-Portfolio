import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

data = pd.read_csv("C:/Users/cjc5n/Documents/Dalhousie/Fundmetrics/Output.csv")
data['donor'] = ['donor' if datapoint <1 else 'non donor' for datapoint in data['maximum_donation']]
data = data.dropna(subset=['twitter_desc'])


def text_process(text):
    removed_punctuation = [char for char in text if char not in string.punctuation]
    removed_punctuation = ''.join(removed_punctuation)
    lemmatizer = WordNetLemmatizer()
    parsed_words = [lemmatizer.lemmatize(word) for word in removed_punctuation.split() if
                    word.lower() not in stopwords.words('english')]

    return parsed_words

random = pd.read_csv('C:/Users/cjc5n/Documents/Dalhousie/Fundmetrics/total_random_users.csv',encoding = "ISO-8859-1")
random = random.rename(columns={"user": "id"})
random['type'] = 'random'

new_donors = data.sample(n=2000, random_state=1)
new_donors = new_donors[['id','twitter_desc']]
new_donors['type'] = 'donor'

merged_dataset = new_donors.append(random,ignore_index=True)

x = merged_dataset['twitter_desc'].apply(str)
y = merged_dataset['type']

# Transform data
tfidf_vectorizer = TfidfVectorizer(use_idf=True, analyzer= text_process)
Encoder = LabelEncoder()

x = tfidf_vectorizer.fit_transform(x)
y = Encoder.fit_transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                    random_state=1)

Naive = MultinomialNB()
Naive.fit(x_train,y_train)
# predict the labels on validation dataset
predictions_NB = Naive.predict(x_test)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, y_test)*100)

# save model
filename = 'finalized_model.sav'
s = pickle.dump(Naive, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

pickle.dump(tfidf_vectorizer.vocabulary_,open("feature.pkl","wb"))
loaded_vec = TfidfVectorizer(vocabulary=pickle.load(open("feature.pkl", "rb")))