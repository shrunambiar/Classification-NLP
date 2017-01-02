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

#using this to develop in iPython
os.chdir("/Users/Morgan/Dropbox/School/NLP/classification_project/NLP_classification_project")

# <codecell>

#Combine all the reviews from all the training files into one dictionary
parsed_reviews={}
for File in os.listdir(os.getcwd()+"/InputPicklefiles"):
    if File[-2:]==".p": #just get pickled files
        parsed_reviews= dict(pickle.load(open("InputPicklefiles/"+File, "rb")).items() + parsed_reviews.items())

# <codecell>

import load_heldout
parsed_reviews_heldout=load_heldout.get()
print parsed_reviews_heldout.items()[1]

# <codecell>

#This part helps debug the imported pickled files
print "# of reviews(lines): "+str(len(parsed_reviews.items()))
print 'here is an example (the first item in the dictionary of "parsed_reviews":'
print parsed_reviews.items()[:5]

# <codecell>

#puts the tags with the words
tagged_words=[]
for sent in parsed_reviews.keys():
    tag=parsed_reviews[sent]
    tagged_words += [(word.lower(),tag) for word in nltk.word_tokenize(sent) if word not in stopwords.words('english')]
print len(tagged_words)

# <codecell>

#extract only the most common words from the training set.
all_words = nltk.FreqDist(w.lower() for w,tag in tagged_words)
word_features = all_words.keys()[:1000]#53.3%
#These word features tried to use larger slices but saw lower accuracy
#they are commented out for explanatory purposes
# word_features = all_words.keys()[:4000]# yielded 51.1% accuracy
# word_features = all_words.keys()[:2000]# yielded 52% accuracy

# <codecell>

#Feature definition function - Features:  #of ADJ
def has_features(sent):
    #Process the sentence here, e.g. tagging the words of sentence
    text = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(text)
    #counts the adjectives in the sentence
    #I originally had just looked for counts of JJ and not JJR and JJS. When
    #I added JJR and JJS the accuracy didn't change significantly.
    
    num_adj = len([tag for (word,tag) in tags if tag in ['JJ','JJR','JJS'] ])
    
    #the words per sentence feature, by its self gets 48.3% accuracy when tested against
    #a portion of the training set, heldout as a test set.
    sent_words=set(sent.split())
    wps = len(sent_words)
    features={}
    
    #adds the top 1000 words as features 
    #53.8 accuracy independent of other features
    for word in word_features:
        features['contains(%s)' % word] = (word in sent_words) 
    #adding # of adjectives
    #51.1 accuracy independent of other features
    features["count adjectives"] = num_adj
    #Add count of words in sentence
    features["words per sentence"]=wps

    
    #The FORMAT of output of this function is a dictionary with labels for the feature as a key
    #and a its corresponding value, with a dictionary item for each feature.
    return features

# <codecell>

featuresets = [(has_features(sent), tag) for (sent,tag) in parsed_reviews.items() and parsed_reviews_heldout.items()]
# for sent,tag in parsed_reviews:

# <codecell>

#test against holdout
#holdout data begins at 3570
train_range=len(parsed_reviews.items())-1 #Against holdout, 63.8% accuracy

# <codecell>

#Set the training range to a set part of the sentences in the training files. 
    #It is also interesting to look at the different results. I got a better
    #accuracy score when I trained it on 50% and then tested on the other 50%.
    #when I test against the heldout data then I will train with 100% of the 
    #training file data.

    
#train_range=int(len(featuresets)*0.1) # this gets 56% accuracy
# train_range=int(len(featuresets)*0.3) #this gets 56% accuracy
# train_range=int(len(featuresets)*0.5) #This gets 57% accuracy
#train_range=int(len(featuresets)*0.7) # this gets 55% accuracy
#train_range=int(len(featuresets)*0.9) # this gets 52% accuracy
#train_range=int(len(featuresets)*0.95) #this gets 52% accuracy

# <codecell>

train_set, test_set = featuresets[train_range:], featuresets[:train_range]
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier = nltk.NaiveBayesClassifier.train(train_set)
print "Accuracy is:"
print nltk.classify.accuracy(classifier, test_set)
print classifier.show_most_informative_features(5)

# <codecell>

##Agenda for our next meeting:
    #We need to come up with a way to test against the hold out

