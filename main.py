from Tools.DatasetPreprocess import LoadData, CleanText
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# Load the IMDB dataframe
df = LoadData(path='Data/IMDB_Dataset.csv')

#converting the sentiments (positive and negatives) to 1 and 0. 
df.sentiment = (df.sentiment.replace({'positive': 1, 'negative': 0})).values

# The dataset is balanced which has been verified from PrintInfoDF function in Tools

# Clean the data
df = CleanText(df, 'review')

# Vectorize the review data
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
vectorized_text = tfidf.fit_transform(df.review)

# Split data, reviews are the inputs, sentiments are the targets
X_train, X_test, y_train, y_test = train_test_split(vectorized_text, df.sentiment, test_size=0.3)

MNB = MultinomialNB()
MNB.fit(X_train, y_train)

accuracy_score = metrics.accuracy_score(MNB.predict(X_test), y_test)
print("accuracy_score = " + str('{:04.2f}'.format(accuracy_score*100))+" %")


