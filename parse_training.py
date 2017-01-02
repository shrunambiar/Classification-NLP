import os
import nltk
import pickle
import random
import re

def parse_reviews(directory,filename):
    f=open(directory+filename)
    lines=f.readlines()
    f.close()
    ##print lines[1].strip()

    parsed_sents={}
    for line in lines:
##        print line
        if line!="[t]\r\n":
            #remove the '\r\n' from the end of each line
            line= line.strip()

            #mark the separation of attributes and sentences
            hashIndex=line.find('##')
            
            #if there are attributes then parse them this way
            if hashIndex>1:            
                sent=line[hashIndex+2:]
                attrString=line[:hashIndex]
                #Converts the string of attributes to list
                #attrs=attrString.split(",")
                m = re.findall('\[([+-])\d\]',attrString)
                if len(set(m)) ==  1:
                    if m[0] == "+": 
                        tag = "pos"
                    else:
                        tag = "neg"
                else:
                    tag = "neutral"
                #    attrObj[attr[:attr.find("[")]]=attr[attr.find("[")+1:-1]
                
            #if there are no attributes, parse them this way           
            else:
                sent=line[2:]
                #attrObj=None
                
            #add to dictionary that keeps track of parsed sentences attributes and values
            parsed_sents[sent]=tag
            
    pickledName=filename[:filename.rfind(".")]+".p"
    pickle.dump(parsed_sents,open(pickledName,"wb"))

trainingDir = "../product_data_training_heldout/training/"
                
for trainingFile in os.listdir(trainingDir):
    if trainingFile != ".DS_Store":
        print trainingFile
        parse_reviews(trainingDir,trainingFile)
