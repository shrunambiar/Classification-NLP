import os
import pickle


def get(path = os.getcwd()):

	os.chdir(path)

	parsed_reviews_heldout={}
	for File in os.listdir(os.getcwd()+"/heldout"):
	    if File[-2:]==".p": #just get pickled files
	        parsed_reviews_heldout= dict(pickle.load(open("heldout/"+File, "rb")).items() + parsed_reviews_heldout.items())
	return parsed_reviews_heldout