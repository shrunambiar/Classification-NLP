import pickle

parsed_reviews = pickle.load( open( "MicroMP3.p", "rb" ) )

print 'here is an example (the first item in the dictionary of "parsed_reviews":'
print parsed_reviews.items()

