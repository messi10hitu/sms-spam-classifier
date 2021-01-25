import pandas as pd

messages = pd.read_csv("C:/Users/HS/PycharmProjects/NLP/7.smsspamcollection/SMSSpamCollection", sep="\t",
                       names=['label', 'message'])  # label = ham and spam.
# print(messages)

import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import PorterStemmer
from nltk import WordNetLemmatizer

stemming = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# sentense = sent_tokenize(message)
# # print(sentense)
corpus = []
# data = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    # print(review)
    review = review.split()
    # print('split: ', review)
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    # print('review', review)
    review = ' '.join(review)
    # print(review)
    corpus.append(review)
# print("lemmatizer: ", corpus)

# instead of Bag of words we use TFiDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(max_features=5000)  # top frequent 5000 columns
X = cv.fit_transform(corpus).toarray()
# print(X)
y = pd.get_dummies(messages['label'])  # the ham and spam converted into 0 or 1.
y = y.iloc[:, 1].values  # 0 specifies ham and 1 specifies spam.

# Train and test split the data:
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=20, random_state=0)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(Xtrain, ytrain)

y_pred = spam_detect_model.predict(Xtest)
print(y_pred)

from sklearn.metrics import confusion_matrix

confusion_m = confusion_matrix(ytest, y_pred)
print(confusion_m)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(ytest, y_pred)
print('accuracy', accuracy)
