# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
from __future__ import division
import os
import nltk
import pickle

from nltk.corpus import stopwords
import re
import random
import math

# <codecell>

os.chdir("/Users/Morgan/Dropbox/School/NLP/classification_project/NLP_classification_project")

# <codecell>

#Combine all the reviews into one dictionary
parsed_reviews={}
# print os.listdir(os.getcwd())
for File in os.listdir(os.getcwd()):
    if File[-2:]==".p": #just get pickled files
        parsed_reviews= dict(pickle.load(open(File, "rb")).items() + parsed_reviews.items())

# <codecell>

print "# of reviews: "+str(len(parsed_reviews.items()))
print 'here is an example (the first item in the dictionary of "parsed_reviews":'
print parsed_reviews.items()[:5]

# <codecell>

#puts the tags with the words
tagged_words=[]
for sent in parsed_reviews.keys():
    tag=parsed_reviews[sent]
    tagged_words += [(word.lower(),tag) for word in nltk.word_tokenize(sent)]
print len(tagged_words)

# <codecell>

all_words = nltk.FreqDist(w.lower() for w,tag in tagged_words)
word_features = all_words.keys()[:2000]
print word_features[:50]

# <codecell>

def has_features(sent):
    #Process the sentence here, e.g. tagging the words of sentence
    text = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(text)
    #counts the adjectives in the sentence
    num_adj = len([tag for (word,tag) in tags if tag =="JJ"])
    sent_words=set(sent.split())
    wps = len(sent_words)
    features={}
    for word in word_features:
        features['contains(%s)' % word] = (word in sent_words)
    #adding # of adjectives
    features["count adjectives"] = num_adj
    #Add count of words in sentence
    features["words per sentence"]=wps
    #Add seed 2000 words
    return features

# <codecell>


# <codecell>

featuresets = [(has_features(sent), tag) for (sent,tag) in parsed_reviews.items()]
# for sent,tag in parsed_reviews:

# <codecell>

train_set, test_set = featuresets[1500:], featuresets[:1500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)

# <codecell>

print classifier.show_most_informative_features(5)

