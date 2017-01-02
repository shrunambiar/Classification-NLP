from __future__ import division
import pickle
import os
import nltk
import math
from nltk.corpus import stopwords
import re


#parsed_reviews = pickle.load( open( "Diaper Champ.p", "rb" ) )
features = {}
rx = re.compile('([&#/(),-])')

def tagcountsfeatures(sent):

	# Replacing ([&#/(),-]) with a space and then replacing multiple spaces with single space
	newsent = re.sub(' +',' ',rx.sub(' ',sent))
	#Tokenize
	text = nltk.word_tokenize(newsent)
	feature={}
	count_past_verbs = 0 
	count_present_verbs = 0
	count_upper = 0
	count_adj = 0
	tagslist = nltk.pos_tag(text)
	#Calculating feature value
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
	feature["Past Tense Verb"]=count_past_verbs
	feature["Present Tense Verb"]=count_present_verbs
	feature["Adjectives"]=count_adj
	feature["Capitalised Words"]=count_upper

	#Logic to calculate tf-idf, it is not imporving accuracy though
	all_words = nltk.FreqDist(w.lower() for w in finallist)
	highestfreq = all_words[all_words.keys()[0]]
	#print "Highest Fequqency word",all_words.keys()[0]
	#print "Higest Frequency",highestfreq
	#print "Length of final list",len(finallist)
	"""
	for w in set(finallist):
		#print w
		#print "Sentence:",newsent
		#print "count in the sentence",newsent.lower().count(w)
		#number of times the word occurs in the sentence
		ctd = newsent.lower().count(w)

		tf = 0.5 + (0.5*ctd)/highestfreq

		#number of times the word occurs in the positive sentences 
		pt = sum([1 for wd in posrev if w.strip() in wd.lower().split()])
		#print "pt:",pt
		p = len(posrev)

		#number of times the word occurs in the negative sentences 
		nt = sum([1 for wd in negrev if w.strip() in wd.lower().split()])
		#print "nt:",nt
		n =len(posrev)

		#number of times the word occurs in the neutral sentences 
		neut = sum([1 for wd in neutralrev if w.strip() in wd.lower().split()])
		#print "neut:",neut
		neu =len(neutralrev)
		idf = math.log((len(lines)/(pt+nt+neut)),2)


		#feature['tf-idf (%s)' % w] = (ctd*math.log((float(nt)/float(pt)),2))
		#feature['tf-idf (%s)' % w] = tf*idf
	"""
		feature['Freq. of (%s) in pos' %(w)]  = pt
		feature["Freq. of (%s) in neg" %(w)]  = nt
		feature["Freq. of (%s) in neut" %(w)]  = neut
		

	return feature

    
    #new = set(normalize(text))
    #all_words = nltk.FreqDist(w.lower() for w in finallist)
    #word_features = all_words.keys()[:2000]
    #for word in word_features:
    	#feature['contains(%s)' % word] = (word in new)

#Normalizing words
def normalize(tokens):
	#print "Reached here normalize"
	rx = re.compile('([&#/(),-])')
	return [rx.sub(' ', t).lower() for t in tokens if t.lower() not in stopwords.words('english') and len(t)>=3 and re.search('^[A-Za-z]+$',t)]
	
#Stemming , but not used currently
def stemming(tokens):
	#print "Reached here stemming"
	lancaster = nltk.LancasterStemmer()
	lancasterlist =  [lancaster.stem(t) for t in tokens]
	return lancasterlist

if __name__ == '__main__':
	pickelFile = "InputPicklefiles"
	lines = list()
	words = list()
	c = 0
	for p in os.listdir(pickelFile):
	    if p != ".DS_Store":
	    	print "---------------------"
	    	print "Now adding:",p
	    	parsed_reviews = pickle.load(open(pickelFile+"/"+p, "rb" ))
	    	# Reading data of all files in a single list
	    	lines =  lines + parsed_reviews.items() 

	#Fetching all the words from the combines list
	words = [re.sub(' +',' ',rx.sub(' ',w)) for l in lines for w in re.sub(' +',' ',rx.sub(' ',l[0])).split()]
	#Normalizing the words
	finallist  = normalize(words)

	#Finding all postive review sentences
	posrev =  [re.sub(' +',' ',rx.sub(' ',w[0])) for w in lines if w[1]=="pos" ]
	#Finding all negative review sentences
	negrev =  [re.sub(' +',' ',rx.sub(' ',w[0])) for w in lines if w[1]=="neg" ]
	#Finding all neutral review sentences
	neutralrev =  [re.sub(' +',' ',rx.sub(' ',w[0])) for w in lines if w[1]=="neutral" ]

	#Creating feature set
	featuresets = [(tagcountsfeatures(sent), orientation) for (sent,orientation) in lines]
	split = int(math.floor(len(featuresets)*0.9))
	train_set, test_set = featuresets[split:], featuresets[:split]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print nltk.classify.accuracy(classifier, test_set)

	    #size = len(featuresets)
	    