'''
Created on Mar 3, 2016

@author: jayasureyar.in@gmail.com
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor , ExtraTreesRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

def count_words(line):
    return len(line.split())

def get_bigrams(string):
    '''
    let me have all the bigrams
    '''
    s = string.lower()
    return [s[i:i+2] for i in xrange(len(s) - 1)]

def string_similarity(str1, str2):
    '''
    perform similarity search 
    '''
    pairs1 = get_bigrams(str1)
    pairs2 = get_bigrams(str2)
    union  = len(pairs1) + len(pairs2)
    hit_count = 0
    for x in pairs1:
        for y in pairs2:
            if x == y:
                hit_count += 1
                break
    return (2.0 * hit_count) / union

def str_stem(s): 
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("'","in.") 
        s = s.replace("inches","in.") 
        s = s.replace("inch","in.")
        s = s.replace(" in ","in. ") 
        s = s.replace(" in.","in.") 
        s = s.replace("''","ft.") 
        s = s.replace(" feet ","ft. ") 
        s = s.replace("feet","ft.") 
        s = s.replace("foot","ft.") 
        s = s.replace(" ft ","ft. ") 
        s = s.replace(" ft.","ft.") 
        s = s.replace(" pounds ","lb. ")
        s = s.replace(" pound ","lb. ") 
        s = s.replace("pound","lb.") 
        s = s.replace(" lb ","lb. ") 
        s = s.replace(" lb.","lb.") 
        s = s.replace(" lbs ","lb. ") 
        s = s.replace("lbs.","lb.") 
        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")   
        s = s.replace(" sq ft","sq.ft. ") 
        s = s.replace("sq ft","sq.ft. ")
        s = s.replace("sqft","sq.ft. ")
        s = s.replace(" sqft ","sq.ft. ") 
        s = s.replace("sq. ft","sq.ft. ") 
        s = s.replace("sq ft.","sq.ft. ") 
        s = s.replace("sq feet","sq.ft. ") 
        s = s.replace("square feet","sq.ft. ") 
        s = s.replace(" gallons ","gal. ") 
        s = s.replace(" gallon ","gal. ") 
        s = s.replace("gallons","gal.") 
        s = s.replace("gallon","gal.") 
        s = s.replace(" gal ","gal. ") 
        s = s.replace(" gal","gal.") 
        s = s.replace("ounces","oz.")
        s = s.replace("ounce","oz.")
        s = s.replace(" oz.","oz. ")
        s = s.replace(" oz ","oz. ")
        s = s.replace("centimeters","cm.")    
        s = s.replace(" cm.","cm.")
        s = s.replace(" cm ","cm. ")
        s = s.replace("milimeters","mm.")
        s = s.replace(" mm.","mm.")
        s = s.replace(" mm ","mm. ")
        s = s.replace("°","deg. ")
        s = s.replace("degrees","deg. ")
        s = s.replace("degree","deg. ")  
        s = s.replace("volts","volt. ")
        s = s.replace("volt","volt. ")
        s = s.replace("watts","watt. ")
        s = s.replace("watt","watt. ")
        s = s.replace("ampere","amp. ")
        s = s.replace("amps","amp. ")
        s = s.replace(" amp ","amp. ")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        s = s.replace("  "," ")
        #s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s.lower()
    else:
        return "null"

def str_common_word(str1, str2):
    #str1 = str1.tolower(str1)str2 = str2.tolower(str2)
    return sum(int(str2.find(word)>=0) for word in str1.split())

def clear_all_non(x):
    return ''.join([i if ord(i) < 128 else ' ' for i in x])

stemmer = SnowballStemmer('english')
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('product_descriptions.csv')
df_pro_attributs = pd.read_csv('attributes.csv')
g = df_pro_attributs.groupby('product_uid').apply(lambda x: x.sum())
df_pro_attributs.value = df_pro_attributs.value.astype('string')
d = df_pro_attributs.groupby('product_uid')['value'].apply(lambda x: "%s" % ' '.join(x))
df_attriutes_merged = pd.DataFrame(d)
df_attriutes_merged['product_uid'] = df_attriutes_merged.index
print df_attriutes_merged.head()

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all,df_attriutes_merged,how='left',on='product_uid')
df_all.value = df_all.value.astype('string')
df_all.rename(columns={'value':'prod_attributes'},inplace=True)
df_all.prod_attributes = df_all.prod_attributes.astype('string')

df_all['prod_attributes'] = df_all['prod_attributes'].map(lambda x:clear_all_non(x))
df_all['prod_attributes'] = df_all['prod_attributes'].map(lambda x:str_stem(x))
print df_all['prod_attributes'][0]

df_all['product_description_1'] = df_all['search_term']+"\t"+df_all['prod_attributes'] +"\t"+ df_all['product_title']
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
#df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)
df_all['word_in_attribute'] = df_all['product_description_1'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all.to_csv('df_all_new_v3.csv',index='False',encoding='ISO-8859-1')

