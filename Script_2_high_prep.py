'''
Created on Mar 3, 2016

@author: user
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor , ExtraTreesRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

def count_words(line):
    return len(line.split())
df = pd.read_csv('df_all_new_v3.csv',encoding='ISO-8859-1')
words_in_title = [count_words(line) for line in df.product_title]
