# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:21:35 2020

@author: Fernando Pereira
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



#Import Data
new_york_tweets = pd.read_json("new_york.json",lines=True)
london_tweets = pd.read_json("london.json",lines=True)
paris_tweets = pd.read_json("paris.json",lines=True)


# How many tweets in NY
print("Num Tweets new york:" + str(len(new_york_tweets)))  #--> 5341
# How many tweets in London
print("Num Tweets London:" + str(len(london_tweets)))  #--> 5341
#How many tweets in Paris
print("Num Tweets Paris:" + str(len(paris_tweets)))   #-->2510


#...................CLASSIFIYNG USING LANGUAGE.................
#Join all Tweets
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()
all_tweets = new_york_text + london_text + paris_text
print("Lenght All Tweets: " + str(len(all_tweets)))
#Join also the labels to compare
labels = [0]*len(new_york_text)+[1]*len(london_text)+[2]*len(paris_text)


#...................MAKING A TRAINING AND TEST SET.................
train_data,test_data,train_labels,test_labels= train_test_split(all_tweets,labels,random_state=1,test_size=0.2)
print("Lenght train data : " + str(len(train_data)))
print("Lenght test data  : " + str(len(test_data)))

#...................MAKING the Count VECTORS.................
counter = CountVectorizer ()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

print("\n")
#print("Train counts")
#print(train_counts[3])
#print("Test counts")
#print(test_counts[3])

#...................TRAIN AND TEST NAIVE BAYES Classifier.................
classifier = MultinomialNB()
classifier.fit(train_counts,train_labels)
predictions = classifier.predict(test_counts)
#print(predictions)

#...................EVALUATING MY MODEL .................
print("Score by  accuracy score : ")
print(accuracy_score(test_labels,predictions))
print("\n")
print("Score by confusion matrix : ")
print(confusion_matrix(test_labels,predictions))

#...................TEST MY TWEET ....................
tweet = "Weather is not good. always raining #Rain"
tweet_counts = counter.transform([tweet])
print("\n")
print("The Tweet: \n" +tweet +" \nWas classified as:\n " + str(classifier.predict(tweet_counts))+"- LONDON")


















