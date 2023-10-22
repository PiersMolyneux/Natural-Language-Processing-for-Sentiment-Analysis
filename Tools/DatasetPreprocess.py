import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from bs4 import BeautifulSoup
import warnings

# For this section https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/notebook used as reference


def LoadData(path):
    # Import dataset
    imdb_df = pd.read_csv(path)
    return imdb_df


def PrintInfoDF(df):
    # Print information on the pandas dataset
    print(df.shape)
    print(df.head(10))
    print(df.describe())
    print(df['sentiment'].value_counts())


# HTML Stripping - NOT WORKING
def strip_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


# Remove any text between square brackets, e.g. [text]
def remove_square_brackets(text):
    # The regular expression '\[[^]]*\]' is used to match and remove content within square brackets.
    # - \[ matches an opening square bracket '['.
    # - [^]]* matches any characters that are not a closing square bracket ']'. This is done with the use of ^ which negates the character class.
    # - \] matches a closing square bracket ']'.
    # Together, '\[[^]]*\]' matches any text within square brackets.
    
    # The re.sub() function replaces any matches with an empty string (''), effectively removing the content within square brackets.
    return re.sub(r'\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    # text = strip_html(text)
    text = remove_square_brackets(text)
    return text

# Remove special characters, e.g. !@,.
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

# Stem the text
def stem_text(text):
    # nltk.download('punkt')
    # nltk.download('stopwords')
    stemmer = nltk.stem.PorterStemmer()
    words = nltk.word_tokenize(text)  # Tokenize the text into words
    stemmed_words = [stemmer.stem(word) for word in words] # stems, rejoins as str
    return ' '.join(stemmed_words)

# Remove Stopwords
def remove_stopwords(text):
    # Tokenize text (alternatively bc of preprocessing could just probs .split)
    tokens = ToktokTokenizer().tokenize(text)
    # Set English stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Clean text by combining above
def CleanText(df, column):
    # Handling panda series, hence use .apply
    df[column] = df[column].apply(denoise_text)
    df[column] = df[column].apply(remove_special_characters)
    df[column] = df[column].apply(stem_text)
    df[column] = df[column].apply(remove_stopwords)
    return df

def CleanString(text):
    text = denoise_text(text)
    text = remove_special_characters(text)
    text = stem_text(text)
    text = remove_stopwords(text)
    return text