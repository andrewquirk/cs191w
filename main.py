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


from newspaper import Article
import newspaper
from numpy import prod
urls = ['http://foxnews.com', 
        'http://breitbart.com', 
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
        'http://occupydemocrats.com',
        'http://abcnews.com',
        'http://msnbc.com']
#urls = ['http://usatoday.com']

# going over articles only mentioning Trump
print("Final Sample/Results: ")
for url in urls: 
    news_source = newspaper.build(url, memoize_articles=False)
    print("test")
    t_d = 0
    for article in news_source.articles:
        ##loading article
        urll = article.url
        #print(urll)
        art = Article(urll)
        art.download()
        art.parse()
        unclean_text = art.text
        
    	#text = normalize_document(art.text)














        #print(text)