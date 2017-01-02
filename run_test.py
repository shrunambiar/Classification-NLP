from __future__ import division
import os
os.chdir("/Users/Morgan/Dropbox/School/NLP/classification_project/NLP_classification_project")
from product_features_morgan import has_features

import pickle



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




def main():
	#retrieve pickled classifier
	classifier=pickle.load(open("classifier.p", "rb"))
	output_file=open("g_3_output.txt","w")
	
	#specify test directory
	testdir= os.getcwd()+"/testset"
	for File in os.listdir(testdir):
		if File !="output.txt" and File !=".DS_Store" and File!=".git":
			print File
			parsed_test_sents=parse_test_reviews(testdir, File)
			# print parsed_test_sents.items()[0]
			print type(parsed_test_sents)
			for i,line in enumerate(parsed_test_sents):
				print line
				if line=="[t]": 
					guess="0"
				else:
					# guess="1"
					guess = classifier.classify(has_features(line))

				output_file.write(File+"\t"+str(i+1)+"\t"+guess+"\n")

	    #product_name \t line_number \t sentiment_polarity \n

	output_file.close()

if __name__ == '__main__':
	main()