################################################
# © 2022 Evan Kimmerrlein and Dawson Whipple
#            All Rights Reserved
################################################

import tweepy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle

# Sentiment
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
SEQUENCE_LENGTH = 300


# Load tokenizer
with open(r'models\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model
model = keras.models.load_model(r'models\sequence.keras')

def get_score(text):
  x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
  return model.predict([x_test])[0][0]

def score_to_sentiment(score):
    if score >= 0.5:
        return POSITIVE
    else:
        return NEGATIVE

# Twitter integration

API_KEY = #"Insert your API key here in quotations"
API_KEY_SECRET = #"Insert your API secret key here in quotations"
BEARER_TOKEN = #"Insert your bearer token key here in quotations"
ACCESS_TOKEN = #"Insert your access token key here in quotations"
ACCESS_TOKEN_SECRET = #"Insert your access token secret key here in quotations"

client = tweepy.Client(
consumer_key= API_KEY, 
consumer_secret= API_KEY_SECRET, 
bearer_token= BEARER_TOKEN, 
access_token=ACCESS_TOKEN, 
access_token_secret=ACCESS_TOKEN_SECRET)

auth = tweepy.OAuthHandler(client.consumer_key, client.consumer_secret)
auth.set_access_token(client.access_token, client.access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


keyword = input("Enter keyword (add '#' to search for hashtags): ")
search_word = keyword + " -rt"
num_of_results = input("How Many Results would you like?: ")

tweets = client.search_recent_tweets(query=search_word, tweet_fields=['created_at'], max_results=num_of_results)

columns = ['Time', 'Tweet', 'Sentiment Analysis', 'Sentiment']
data = []
sum = 0
count = 0
score = 0

for tweet in tweets.data:
    score = get_score(tweet.text)
    sum += score
    #print(tweet.text + ": " + str(get_score(tweet.text)))
    data.append([tweet.created_at, tweet.text, score, score_to_sentiment(tweet.text)])
    count = count + 1

average = (sum/count)
df = pd.DataFrame(data, columns=columns)
print("Average Sentiment of " + str(keyword) + ": " + str(average))