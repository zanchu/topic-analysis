
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from gensim import corpora, models
from stop_words import get_stop_words
import gensim
import csv
import pandas as pd
import string
import argparse

# define input argument - file name
parser = argparse.ArgumentParser(description='Run LDA and LSI with csv file contain ratings and comments')
parser.add_argument('-f', help='file name', required=True)
args = vars(parser.parse_args())
file = args['f']
print('file name: ',file)

# initialize tokenizer
tokenizer = TweetTokenizer()

# detractors is defined as NPS with rating 1-3
def isDetractor(rating):  
    return rating == '1' or doc[0] == '2' or doc[0] == '3'

# create English stop words list, add punctuation to stopword
en_stop = get_stop_words('en')
for i in (string.punctuation):
    en_stop.append(i)

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

data =[]
# local csv file contains NPS rating and comments
# File format: First column contains NPS rating, '1' - '5'; Second column contains NPS comment text
# Sample row: '3','Good design overall!'
#file = "CommentRating.csv"

#read cvs file line by line as tuple (rating, comments)
with open(file, "r") as f:    
    data=[tuple(line) for line in csv.reader(f)]

# list for tokenized documents in loop
texts = []

# loop through document list and retain only detractors' comments for analysis
for doc in data:
    
    if not isDetractor(doc[0]):
        continue
        
    i = doc[1]
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stopped_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

#parameters for topic modeling
model_num_topics = 5
model_pass_lda = 2
model_topic_word = 3


#generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=model_num_topics, passes=model_pass_lda)
#print topics
print('---Top ',model_num_topics,' topics with LDA with ',model_topic_word,' words:')
for top in ldamodel.print_topics(num_topics=model_num_topics, num_words=model_topic_word):
  print (top)


#generate LSI model
lsi = gensim.models.lsimodel.LsiModel(corpus, id2word=dictionary, num_topics=model_num_topics)
#print topics
print('---Top ',model_num_topics,' topics with LSI with ',model_topic_word,' words:')
for top in lsi.print_topics(num_topics=model_num_topics, num_words=model_topic_word):
  print (top)

