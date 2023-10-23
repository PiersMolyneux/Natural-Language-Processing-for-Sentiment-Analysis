# Natural Language Processing for Sentiment Analysis

**"Natural Language Processing for Sentiment Analysis"** is a project that utilizes machine learning, specifically a Naive Bayes classifier, to analyze and classify text data based on sentiment, specifically for binary classification of positive or negative sentiment. It's built using FastAPI for deployment.

## Objective

The primary objective of this project is to develop a sentiment analysis system that can automatically classify text data (e.g., social media comments, product reviews, customer feedback) into sentiment categories, namely positive or negative. This was a learning project.

## Steps used in this Project, including potential steps if optimizing the model was a focus.

### 1. Data Collection
Collect a dataset of text data that includes labeled examples of sentiment, where each piece of text is associated with a sentiment label (positive or negative). The dataset chosen is the IMDb movie reviews dataset.

### 2. Data Preprocessing
Preprocess the text data, as well as analyse to make sure it is suitable for training. Preprocess may involve tasks like lowercasing, tokenization, and removing stopwords, punctuation, and special characters. Text data often needs to be cleaned and standardized for analysis. 

### 3. Feature Extraction
Convert the text data into numerical features suitable for machine learning. Common methods include using techniques like TF-IDF (Term Frequency-Inverse Document Frequency), or bag of words, or word embeddings such as Word2Vec, GloVe.

### 4. Model Selection
Choose a Naive Bayes classifier as the sentiment analysis model for binary classification of positive or negative sentiment. Train the model on the preprocessed and feature-extracted data. Other models are possible, such as Random Forest, or Regression models.

### 5. Model Evaluation
Split dataset into training and testing sets and evaluate the Naive Bayes model's performance for binary sentiment classification. Common evaluation metrics for sentiment analysis include accuracy, precision, recall, and F1-score. The results could also be visualized with a confusion matrix.

### 6. Model Interpretability
Analyze feature importance to understand which words or features have the most significant influence on sentiment classification.

### 7. Fine-Tuning
Experiment with hyperparameter tuning to optimize the Naive Bayes model's performance.

### 8. Deployment
Once the sentiment analysis model performs well, deploy it using FastAPI.


## Skills Gained in this project

- Text data preprocessing and feature extraction.
- Building and deploying a Naive Bayes classifier for binary sentiment classification.
- Understanding of NLP and machine learning techniques.
- Experience launching a ML model with FastAPI.
