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
import time



from newspaper import Article
import newspaper
from numpy import prod
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

# urls = ['http://www.wsj.com',
# 'http://usatoday.com', 
# 'http://newyorktimes.com'
# ]

#urls = ['https://www.aljazeera.com/topics/regions/europe.html']
#urls = ['https://www.bbc.com/news/world/europe']


sent = "this is a test sentence"
# sid = SentimentIntensityAnalyzer()
# ss = sid.polarity_scores(sent)
# neg = ss['pos']
# print(neg)
# print(ss)
# print("ok")


for url in urls: 
    neg = 0.0
    pos = 0.0
    news_source = newspaper.build(url, memoize_articles=False)
    counter = 0.0
    print("url " + str(url) + " has this many articles: " + str(len(news_source.articles)))

    if (len(news_source.articles) > 50):

        feature_size = 300   # Word vector dimensionality  
        window_context = 30    # Context window size                                                                                    
        min_word_count = 1   # Minimum word count                        
        sample = 1e-3    # Downsample setting for frequent words


        tokens = []

        for article in news_source.articles:
            ##loading article
            urll = article.url
            #print(urll)
            art = Article(urll)
            art.download()
            art.parse()
            #print(art.text)
            unclean_text = art.text
        # #     ref_bool = False
            tokens.append(nltk.word_tokenize(unclean_text))
        print("beginning training")
        model = word2vec.Word2Vec(tokens, size=feature_size, window=window_context, min_count=min_word_count, sample=sample, iter=50)
        #print(model.most_similar('good', 10))
        print(model.wv['refugee'])
        print(model.wv['human'])



    #     tokens = set(tokens)
    #     #print(tokens)
    #     if 'refugee' in tokens or 'Refugee' in tokens or 'refugees' in tokens or 'Refugees' in tokens  or 'immigrant' in tokens or 'Immigrant' in tokens or 'immigrants' in tokens or 'Immigrants' in tokens:
    #         print(unclean_text)
    #     for word in tokens:
    #         #print(word)
    #         if (word == "refugee" or word == "Refugee" or word == "Migrant" or word == "migrant"):
    #             ref_bool = True
        
    #     if ref_bool:
    #         sid = SentimentIntensityAnalyzer()
    #         ss = sid.polarity_scores(unclean_text)
    #         neg += ss['neg']
    #         pos += ss['pos']
    #         counter += 1.0
    # if counter > 0:
    #     print("The scores are for " + url)
    #     print("Positivity: " + str(pos/counter))
    #     print("Negativity: " + str(neg/counter))













