from __future__ import division
import pickle
import os
import nltk
import math
from nltk.corpus import stopwords
import re
import random
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk.collocations
import nltk.corpus
import collections
import string


#parsed_reviews = pickle.load( open( "Diaper Champ.p", "rb" ) )
rx = re.compile('([&#/(),-])')
all_words = nltk.FreqDist()

#This method uses normalization
def features(sent):
    features = {}

    porter = nltk.PorterStemmer()
    sent = sent.translate(string.maketrans("",""), string.punctuation).lower().strip().split()
    sent_stem = [porter.stem(t) for t in sent]
    
    count_past_verbs = 0 
    count_present_verbs = 0
    count_upper = 0
    count_adj = 0
    count_positive_pair = 0
    count_negation = 0
    count_negation_pair = 0

    #normalizing
    text = normalize(sent)
    #stemming
    
    tagslist = nltk.pos_tag(text)

    for t in tagslist:
        if t[1] in ['VBD','VBN']:
            count_past_verbs+=1
        if t[1] in ['VBG','VBZ']:
            count_present_verbs+=1
        if (t[0].isupper() and len(t[0]) > 3 and t[1] not in ['NN','NNP','NNPS','NNS','PRP','PRP$']):
            count_upper+=1
        if t[1] in ['JJ','JJR','JJS']:
            count_adj+=1

    #Assigning feature value
    
    features["Past Tense Verb"]=count_past_verbs
    features["Present Tense Verb"]=count_present_verbs
    features["Adjectives"]=count_adj
    features["Capitalised Words"]=count_upper
    
    #Calculating feature value
    
    for word in pos_top_vrb_s:
        features["has %s" % word] = word in sent_stem
    #for word in neg_top_vrb_s:
        #features["has %s" % word] = word in sent_setm
    for word in neutral_top_vrb_s:
        features["has %s" % word] = word in sent_stem
    for word in pos_top_adj_s:
        features["has %s" % word] = word in sent_stem
    for word in neutral_top_adj_s:
        features["has %s" % word] = word in sent_stem
    

    #Derek's
    for (w1,t1), (w2,t2) in nltk.bigrams(tagslist):
                # feature2: count positive word pair
        if (w1.lower(),w2.lower()) in [('no','problems'),
                                    ('no','problem'),
                                    ('no','big'),
                                    ('no','trouble'),
                                    ('no','complaints'),
                                    ('no','issues'),
                                    ('no','other'),
                                    ('no','odor'),
                                    ('not','difficult'),
                                    ('highly','recommended'),
                                    ('I','recommended'),
                                    ('well','worth'),
                                    ('is','worth'),
                                    ('was','worth'),
                                    ('\'s','worth'),
                                    ('centainly','worth'),
                                    ('definitely','worth'),
                                    ('indeed','worth'),
                                    ('it','worth'),                                                                                
                                    ]:
            count_positive_pair +=1
        if w1.lower() in ['not','never','no','wrong','junk'] and (t2 in ['JJ','JJR','JJS'] or t2 in ['VB','VBD','VBG','VBN','VBP','VBZ'] or t2 in ['NN','NNS']):                
            count_negation +=1
        if (w1.lower(),w2.lower()) in [('no','way'),
                                        ('not','work'),
                                        ('not','compatible'),
                                        ('not','worth'),
                                        ('not','possible'),
                                        ('not','find'),
                                        ('no','manual'),
                                        ('not','allow'),
                                        ('not','buying'),
                                        ('not','easy'),
                                        ('never','get'),
                                        ('stay','away'),
                                        ('not','recommended'),
                                        ]:
            count_negation_pair +=1

    features["Positive Pair"]=count_positive_pair
    features["Negation Count"]= count_negation 
    features["Negation Pair"]=count_negation_pair

    #Morgan's
    num_adj = len([tag for (word,tag) in tagslist if tag in ['JJ','JJR','JJS'] ])
    
    #the words per sentence feature, by its self gets 48.3% accuracy when tested against
    #a portion of the training set, heldout as a test set.
    sent_words=set(sent)
    wps = len(sent_words)
    
    #adds the top 1000 words as features 
    #53.8 accuracy independent of other features
    for word in word_features:
        features['contains(%s)' % word] = (word in sent_words) 
    #adding # of adjectives
    #51.1 accuracy independent of other features
    features["count adjectives"] = num_adj
    #Add count of words in sentence
    features["words per sentence"]=wps

    return features
    

#Normalizing words
def normalize(tokens):
    #print "Reached here normalize"
    #rx = re.compile('([&#/(),-])')
    return [t.translate(None, string.punctuation).lower() for t in tokens if t.lower() not in stopwords.words('english') and len(t)>=3]
    
#Stemming , but not used currently
def stemming(tokens):
    #print "Reached here stemming"
    lancaster = nltk.LancasterStemmer()
    lancasterlist =  [lancaster.stem(t) for t in tokens]
    return lancasterlist

def get_seed_list_stemming(reviews):
    porter = nltk.PorterStemmer()
    neutral_adj = []
    pos_adj = []
    neg_adj = []
    neutral_vrb = []
    pos_vrb = []
    neg_vrb = []
    #stopwords = nltk.corpus.stopwords.words('english')
    
    for sent,label in reviews:
        clean_sent = sent.translate(string.maketrans("",""), string.punctuation).lower().strip().split()
        #clean_sent_no_sw = [w for w in clean_sent if w not in stopwords]     ##Removing stop words before tagging does not work.
        tagged = nltk.pos_tag(clean_sent)
        if label == 'neutral':
            for i,j in tagged:
                if j[0] == 'J':
                    neutral_adj.append(porter.stem(i))
                if j[0] == 'V':
                    neutral_vrb.append(porter.stem(i))
        if label == 'pos':
            for i,j in tagged:
                if j[0] == 'J':
                    pos_adj.append(porter.stem(i))
                if j[0] == 'V':
                    pos_vrb.append(porter.stem(i))
        if label == 'neg':
            for i,j in tagged:
                if j[0] == 'J':
                    neg_adj.append(porter.stem(i))
                if j[0] == 'V':
                    neg_vrb.append(porter.stem(i))

    return [neutral_adj, neutral_vrb, pos_adj, pos_vrb, neg_adj, neg_vrb]

def parse_test_reviews(directory, filename):    
    f = open(directory +"/"+ filename)
    lines=f.readlines()
    f.close()
    parsed_sents=[]
    for line in lines:
        #remove the '\r\n' from the end of each line
        line_stripped= line.strip()

        sent=line_stripped.split('\t')[1]
        if sent.startswith("[t]"):
            parsed_sents.append(sent)
        else:
            parsed_sents.append(sent[2:])        
    return parsed_sents


if __name__ == '__main__':
    pickelFile = "training"
    heldoutFiles = "testset"
    lines = list()
    heldoutlines = list()
    words = list()
    c = 0
    for p in os.listdir(pickelFile):
        if p != ".DS_Store":
            print "Now adding:",p
            parsed_reviews = pickle.load(open(pickelFile+"/"+p, "rb" ))
            # Reading data of all files in a single list
            print len(parsed_reviews.items())
            lines = lines + parsed_reviews.items()
    random.shuffle(lines)

    """
    #Heldout data
    print "Adding test files"
    for p in os.listdir(heldoutFiles):
        if p != ".DS_Store":
            print "Now adding:",p
            parsed_reviews_1 = pickle.load(open(heldoutFiles+"/"+p, "rb" ))
            # Reading data of all files in a single list
            print len(parsed_reviews_1.items())
            heldoutlines = heldoutlines + parsed_reviews_1.items()
    random.shuffle(heldoutlines)
    """

    #Getting the training and test data
    train = lines
    #test = heldoutlines

    tagged_words=[]
    for sent,tag in train:
        tagged_words += [(word.lower(),tag) for word in nltk.word_tokenize(sent) if word not in stopwords.words('english')]

    all_words = nltk.FreqDist(w.lower() for w,tag in tagged_words)
    word_features = all_words.keys()[:1000]


    #Fetching seed words
    seed_words = get_seed_list_stemming(train)

    neutral_top_adj_s = nltk.FreqDist(seed_words[0]).keys()
    pos_top_adj_s = nltk.FreqDist(seed_words[2]).keys()
    neg_top_adj_s = nltk.FreqDist(seed_words[4]).keys()

    neutral_top_vrb_s = nltk.FreqDist(seed_words[1]).keys()
    pos_top_vrb_s = nltk.FreqDist(seed_words[3]).keys()
    neg_top_vrb_s = nltk.FreqDist(seed_words[5]).keys()


    #Not required for testing
    train_set= [(features(sent), orientation) for (sent,orientation) in train]
    #test_set= [(features(sent), orientation) for (sent,orientation) in test]

     #Not required for testing
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    #print "tags classification"
    #print nltk.classify.accuracy(classifier, test_set)

    #Not required for testing
    pickle.dump(classifier,open("classifier.p","wb"))

    classifier=pickle.load(open("classifier.p", "rb"))
    output_file=open("g_3_output.txt","w")
    
    #specify test directory
    testdir= os.getcwd()+"/testset"
    for File in os.listdir(testdir):
        if File !="output.txt" and File !=".DS_Store" and File!=".git":
            print File
            parsed_test_sents=parse_test_reviews(testdir, File)
            # print parsed_test_sents.items()[0]
            for i,line in enumerate(parsed_test_sents):
                if line=="[t]": 
                    sentiment="0"
                else:
                    # guess="1"
                    guess = classifier.classify(features(line))
                    if guess=="pos":
                        sentiment="+1"
                    if guess=="neg":
                        sentiment="-1"
                    if guess=="neutral":
                        sentiment="0"

                output_file.write(File+"\t"+str(i+1)+"\t"+sentiment+"\n")
    output_file.close()


