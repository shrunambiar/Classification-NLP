import os
import re
import pickle

def parse_reviews(directory, filename):    
    f = open(trainingDir + filename)
    lines=f.readlines()
    f.close()
    parsed_sents={}
    for line in lines:
        if line!="[t]\n":
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
                    
                #if there are no attributes, parse them this way           
                else:
                    sent=line[2:]
                    tag = "neutral"

                parsed_sents[sent]=tag
        

        pickledName=filename[:filename.rfind(".")]+".p"
        pickle.dump(parsed_sents,open("heldout/"+pickledName,"wb"))
                

trainingDir = "../product_data_training_heldout/heldout/"
os.listdir(trainingDir)

if __name__ == '__main__':
    for File in os.listdir(trainingDir):
        if File != ".DS_Store":
            print File
            parse_reviews(trainingDir,File)

