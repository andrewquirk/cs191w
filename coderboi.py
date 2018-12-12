#import newspaper
#from keras.models import Sequential
#import keras
import re
import os
import nltk
from gensim.models import word2vec
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cosine, euclidean, jaccard
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
pd.options.display.max_colwidth = 200
from string import punctuation


# creates an empty list of dicts to hold the speeches of each congress
agg = [{'R':[], 'D':[]} for _ in range(10)]

# returns the session of congress mapping to that specific year
def get_sesh(year):
    return int((year-0.5)/2) - 893

# returns the index in the agg list of speeches from a given year
def get_i(year):
    return get_sesh(year) - 103

# gets the year as an integer from a full date string
def get_year(date):
    return int(re.findall(r"\D(\d{4})\D", ' ' + date + ' ')[0])

# gets only the first and last word of a person's name to match transcript format
def trim_name(name):
    name_list = name.split()
    return ' '.join([name_list[0], name_list[-1]])

# gets the properly formatted name of the speaker of a given speech
def get_speaker_name(speech):
    name = speech['speaker']['name']
    return name['first'] + ' ' + name['last'] 

# for the entire dataset, run for i in range(1, 2216)
# for i in range(1, 2707):
#     filename = '../../tagged_transcripts/house_hearing_transcript'+str(i)+'.json'
#     with open(filename) as json_data:
#         data = json.load(json_data) 
#         # First, load in the poltical affiliations of everyone listed in the people header
#         affiliations = {}
#         for person in data['people']:
#             name = str(person['name']['first'] + ' ' + person['name']['last'])
#             affiliations[trim_name(name)] = person['party']
#         # Next, load everything that was said for each party into the overall dict
#         agg_index = get_i(get_year(data['date_aired']))
#         for speech in data['transcript']:
#             aff = affiliations.get(get_speaker_name(speech))
#             # only add what was said if the affiliation of the speaker is R/D
#             if aff == 'R' or aff == 'D':
#                 agg[agg_index][aff].append(speech['speech'])

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

end_r_speeches =[]
# for year in range(9):
# for speech in agg[9]['R']:
#     news = normalize_document(speech)
#     end_r_speeches.append(news)
from newspaper import Article
import newspaper

rep_sauce = newspaper.build('http://breitbart.com', memoize_articles=False)
for article in rep_sauce.articles:
    urll = article.url
    art = Article(urll)
    art.download()
    art.parse()
    text = normalize_document(art.text)
    end_r_speeches.append(text)

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in end_r_speeches]

# Set values for various parameters
feature_size = 100   # Word vector dimensionality  
window_context = 30    # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3    # Downsample setting for frequent words

r_w2v_model= word2vec.Word2Vec(tokenized_corpus, size=feature_size, window=window_context, min_count=min_word_count, sample=sample, iter=50)

end_d_speeches =[]
# for year in range(9):
dem_sauce = newspaper.build('http://occupydemocrats.com', memoize_articles=False)
for article in dem_sauce.articles:
    urll = article.url
    art = Article(urll)
    art.download()
    art.parse()
    text = normalize_document(art.text)
    end_d_speeches.append(text)


# for speech in agg[9]['D']:
#     news = normalize_document(speech)
#     end_d_speeches.append(news)

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in end_d_speeches]

# Set values for various parameters
feature_size = 100   # Word vector dimensionality  
window_context = 30    # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3    # Downsample setting for frequent words

d_w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size, window=window_context, min_count=min_word_count, sample=sample, iter=50)


from numpy import prod

urls = ['http://foxnews.com', 
        'http://economist.com', 
        'http://newyorktimes.com',
        'http://www.wsj.com',
        'http://www.huffingtonpost.com',
        'http://www.motherjones.com',
        'http://www.newyorker.com',
        'http://reuters.com',
        'http://usatoday.com',
        'http://npr.org',
        'http://ap.org',
        'http://abcnews.com',
        'http://msnbc.com']

# going over articles only mentioning Trump
print("Doing larger sample")
for url in urls: 
    news_source = newspaper.build(url, memoize_articles=False)
    diff = 0

    assign_r = []
    assign_d = []
    t_r = 0
    t_d = 0
    for article in news_source.articles:

    	##loading article

    	urll = article.url
    	art = Article(urll)
    	art.download()
    	art.parse()
    	text = normalize_document(art.text)
    	# if (("trump" not in urll) and ("politics" not in urll)):
    	# 	continue

    	##testing
    	b_vec = text.split(" ") 
    	total_d = 0
    	counter_d = 0
    	counter_r = 0
    	total_r = 0
    	inn=0
    	out=0
    	for word in b_vec:
    	    if(word in d_w2v_model.wv):
                arr = d_w2v_model.wv[word]
                total_d += (sum(arr) / len(arr))
    	    if(word in r_w2v_model.wv):
                arr = r_w2v_model.wv[word]
                total_r += (sum(arr) / len(arr))
    	if (total_r > total_d):
    		assign_r.append(url)
    	else:
    		assign_d.append(url)
    	diff += abs(total_r - total_d)
    	t_d += total_d
    	t_r += total_r

    print(url)
    print('The overall difference in the score is ' + str(diff))
    print('The number of articles classified as R: ' + str(len(assign_r)))
    print('The number of articles classified as D: '+ str(len(assign_d)))
    print('The total R score is: ' + str(t_r))
    print('The total D score is: ' + str(t_d))
    print(' ')
    print(' ')


print("Doing politics")
for url in urls: 
    news_source = newspaper.build(url, memoize_articles=False)
    diff = 0

    assign_r = []
    assign_d = []
    t_r = 0
    t_d = 0
    for article in news_source.articles:

        ##loading article

        urll = article.url
        art = Article(urll)
        art.download()
        art.parse()
        text = normalize_document(art.text)
        if ("trump" not in urll):
          continue

        ##testing
        b_vec = text.split(" ") 
        total_d = 0
        counter_d = 0
        counter_r = 0
        total_r = 0
        inn=0
        out=0
        for word in b_vec:
            if(word in d_w2v_model.wv):
                arr = d_w2v_model.wv[word]
                total_d += (sum(arr) / len(arr))
            if(word in r_w2v_model.wv):
                arr = r_w2v_model.wv[word]
                total_r += (sum(arr) / len(arr))
        if (total_r > total_d):
            assign_r.append(url)
        else:
            assign_d.append(url)
        diff += abs(total_r - total_d)
        t_d += total_d
        t_r += total_r

    print(url)
    print('Only articles containing the string "Trump" or "Politics"')
    print('The overall difference in the score is ' + str(diff))
    print('The number of articles classified as R: ' + str(len(assign_r)))
    print('The number of articles classified as D: '+ str(len(assign_d)))
    print('The total R score is: ' + str(t_r))
    print('The total D score is: ' + str(t_d))
    print(' ')
    print(' ')

# #---------------------------------------------------------------------------    
# # going over all articles, not necessarily political ones
# print("Doing avg2")
# for url in urls: 
#     news_source = newspaper.build(url, memoize_articles=False)
#     diff = 0
#     assign_r = []
#     assign_d = []
#     t_r = 0
#     t_d = 0
#     for article in news_source.articles:

#         ##loading article

#         urll = article.url
#         art = Article(urll)
#         art.download()
#         art.parse()
#         text = normalize_document(art.text)

#         ##testing
#         b_vec = text.split(" ") 
#         total_d = 0
#         counter_d = 0
#         counter_r = 0
#         total_r = 0
#         inn=0
#         out=0
#         for word in b_vec:
#             if(word in d_w2v_model.wv):
#                 arr = d_w2v_model.wv[word]
#                 total_d += (sum(arr) / len(arr))
#             if(word in r_w2v_model.wv):
#                 arr = r_w2v_model.wv[word]
#                 total_r += (sum(arr) / len(arr))

#         if (total_r > total_d):
#             assign_r.append(url)
#         else:
#             assign_d.append(url)
#         diff += abs(total_r - total_d)
#         t_d += total_d
#         t_r += total_r

#     print(url)
#     print('The overall difference in the score is ' + str(diff))
#     print('The number of articles classified as R: ' + str(len(assign_r)))
#     print('The number of articles classified as D: '+ str(len(assign_d)))
#     print('The total R score is: ' + str(t_r))
#     print('The total D score is: ' + str(t_d))


# #---------------------------------------------------------------------------    
# # going over all articles, not necessarily political ones
# print("Doing product")
# for url in urls: 
#     news_source = newspaper.build(url, memoize_articles=False)
#     diff = 0
#     assign_r = []
#     assign_d = []
#     t_r = 0
#     t_d = 0
#     for article in news_source.articles:

#         ##loading article

#         urll = article.url
#         art = Article(urll)
#         if ("trump" not in urll):
#             continue
#         art.download()
#         art.parse()
#         text = normalize_document(art.text)

#         ##testing
#         b_vec = text.split(" ") 
#         total_d = 0
#         counter_d = 0
#         counter_r = 0
#         total_r = 0
#         inn=0
#         out=0
#         for word in b_vec:
#             if(word in d_w2v_model.wv):
#                 arr = d_w2v_model.wv[word]
#                 total_d += prod(arr)
#             if(word in r_w2v_model.wv):
#                 arr = r_w2v_model.wv[word]
#                 total_r += prod(arr)

#         if (total_r > total_d):
#             assign_r.append(url)
#         else:
#             assign_d.append(url)
#         diff += abs(total_r - total_d)
#         t_d += total_d
#         t_r += total_r

#     print(url)
#     print('The overall difference in the score is ' + str(diff))
#     print('The number of articles classified as R: ' + str(len(assign_r)))
#     print('The number of articles classified as D: '+ str(len(assign_d)))
#     print('The total R score is: ' + str(t_r))
#     print('The total D score is: ' + str(t_d))

#---------------------------------------------------------------------------    
# going over all articles, not necessarily political ones
# print("Doing not just politics")
# for url in urls: 
#     news_source = newspaper.build(url, memoize_articles=False)
#     diff = 0
#     assign_r = []
#     assign_d = []
#     t_r = 0
#     t_d = 0
#     for article in news_source.articles:

#         ##loading article

#         urll = article.url
#         art = Article(urll)
#         art.download()
#         art.parse()
#         text = normalize_document(art.text)

#         ##testing
#         b_vec = text.split(" ") 
#         total_d = 0
#         counter_d = 0
#         counter_r = 0
#         total_r = 0
#         inn = 0
#         out = 0
#         for word in b_vec:
#             if(word in d_w2v_model.wv):
#                 arr = d_w2v_model.wv[word]
#                 total_d += int(prod(arr))
#             if(word in r_w2v_model.wv):
#                 arr = r_w2v_model.wv[word]
#                 total_r += int(prod(arr))

#         if (total_r > total_d):
#             assign_r.append(url)
#         else:
#             assign_d.append(url)
#         diff += abs(total_r - total_d)
#         t_d += total_d
#         t_r += total_r

#     print(url)
#     print('The overall difference in the score is ' + str(diff))
#     print('The number of articles classified as R: ' + str(len(assign_r)))
#     print('The number of articles classified as D: '+ str(len(assign_d)))
#     print('The total R score is: ' + str(t_r))
#     print('The total D score is: ' + str(t_d))
