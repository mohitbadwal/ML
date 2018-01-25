##############################################################################
#import statements
##############################################################################
import sys
from sklearn import learning_curve
import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import cross_validation
from sklearn import svm
from datetime import timedelta
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import load_digits
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
import re
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.preprocessing import Imputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import os
import seaborn as sb
import string

###############################################################################
#input csv
###############################################################################
if len(sys.argv) == 3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
else:
    print("Format python filename train_dataset_path test_dataset_path")
    sys.exit(1)


df_c = pd.read_csv(train_file,
                      encoding='cp1256')
df_c_test = pd.read_csv(test_file,
                      encoding='cp1256')
df_test=df_c_test
df_c=df_c[df_c['page_pageType']=='REMITTANCE_PAGE']
df_test=df_test[df_test['page_pageType']=='REMITTANCE_PAGE']



###############################################################################
#functions
###############################################################################

##################################
#cleaning data
##################################
def clean_data(df_c):
    df_c['row_string'] = df_c['row_string'].apply(lambda x: x.lower())
    df_c['row_string_new'] = df_c['row_string'].str.replace('[^\w\s]', ' ')

    # df_c['row_string_new']=df_c['row_string'].apply(lambda x: ''.join([i for i in x if i not in punc]))
    # df_c['row_string_new'] = df_c['row_string_new'].str.replace('\d+', '')
    stop = set(stopwords.words('english'))

    df_c['row_string_new'] = df_c['row_string_new'].apply(lambda x: [item for item in x.split() if item not in stop])

    df_c['row_string_new'] = df_c['row_string_new'].apply(lambda x: ' '.join([i for i in x]))


    return df_c


##################################
#alpha_numeric_ratio
##################################

def alpha_func(x):
    digit=sum([c.isdigit() for c in x])
    alpha=sum([c.isalpha() for c in x])
    k=0.0
    if(alpha==0):
        k=1
    elif((float(digit)/float(alpha))<0.05):
        k=0
    else:
        k=1

    return k


##################################
#distance_from_top
##################################

def distance_from_top(x):
    if(x<=30):
        return 0
    else:
        return 1


##################################
#vocabulary
##################################


def func(df_c):
    df_c['description gross'] = None
    df_c['discount payment'] = None
    df_c['amt balance'] = None
    df_c['balance due'] = None
    df_c['date type'] = None
    df_c['due discount'] = None
    df_c['invoice description'] = None
    df_c['original amt'] = None
    df_c['reference original'] = None
    df_c['type reference'] = None
    df_c['date amount'] = None
    df_c['amount deduction'] = None
    df_c['invoice inv'] = None
    df_c['gross discount'] = None
    df_c['invoice gross'] = None
    df_c['deductions amount'] = None
    df_c['gross net'] = None
    df_c['amount discounts'] = None
    df_c['description amount'] = None
    df_c['discounts amount'] = None
    df_c['gross'] = None
    df_c['invoice loc'] = None
    # df_c['grand tot'] = None
    df_c['loc invoice'] = None
    df_c['memo invoice'] = None
    df_c['number amount'] = None
    df_c['date invoice'] = None
    df_c['discount net'] = None
    df_c['number date']=None

    for i in range(0, df_c.shape[0]):


        if (re.search('[a-z0-9]*((description(\s)?gros)|(desc[a-z0-9]*(\s)?gros))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['description gross'].values[i] = 0
        else:
            df_c['description gross'].values[i] = 1

        if (re.search('[a-z0-9]*((discount(\s)?pay)|(disc[a-z0-9]*(\s)?pay))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['discount payment'].values[i] = 0
        else:
            df_c['discount payment'].values[i] = 1

        if (re.search('[a-z0-9]*((amt(\s)?balance)|(amt(\s)bal)|(amount(\s)?balance))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['amt balance'].values[i] = 0
        else:
            df_c['amt balance'].values[i] = 1

        if (re.search('[a-z0-9]*((balance(\s)?due)|(bal[a-z0-9]*(\s)?due))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['balance due'].values[i] = 0
        else:
            df_c['balance due'].values[i] = 1

        if (re.search('[a-z0-9]*((date(\s)?type))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['date type'].values[i] = 0
        else:
            df_c['date type'].values[i] = 1

        if (re.search('[a-z0-9]*((due(\s)?discount)|(due(\s)?disc))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['due discount'].values[i] = 0
        else:
            df_c['due discount'].values[i] = 1

        if (re.search('[a-z0-9]*((invoice(\s)?desc)|(inv[a-z0-9]*(\s)?desc))[a-z0-9]*',df_c['row_string_new'].values[i]) == None):
            df_c['invoice description'].values[i] = 0
        else:
            df_c['invoice description'].values[i] = 1


        if (re.search('[a-z0-9]*((original(\s)?amt)|(orig[a-z0-9]*(\s)?amou))[a-z0-9]*',df_c['row_string_new'].values[i]) == None):
            df_c['original amt'].values[i] = 0
        else:
            df_c['original amt'].values[i] = 1

        if (re.search('[a-z0-9]*((refer(\s)?orig))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['reference original'].values[i] = 0
        else:
            df_c['reference original'].values[i] = 1

        if (re.search('[a-z0-9]*((type(\s)?ref)|(ref[a-z0-9]*(\s)?type))[a-z0-9]*',df_c['row_string_new'].values[i]) == None):
            df_c['type reference'].values[i] = 0
        else:
            df_c['type reference'].values[i] = 1

        if (re.search('[a-z0-9]*((dat[a-z0-9]*(\s)?amou))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['date amount'].values[i] = 0
        else:
            df_c['date amount'].values[i] = 1

        if (re.search('[a-z0-9]*((amou[a-z0-9]*(\s)?deduc)| (deduc[a-z0-9]*(\s)?amt))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['amount deduction'].values[i] = 0
        else:
            df_c['amount deduction'].values[i] = 1

        if (re.search('[a-z0-9]*((inv[a-z0-9]*(\s)?inv))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['invoice inv'].values[i] = 0
        else:
            df_c['invoice inv'].values[i] = 1

        if (re.search('[a-z0-9]*(gro[a-z0-9]*(\s)?disc)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['gross discount'].values[i] = 0
        else:
            df_c['gross discount'].values[i] = 1

        if (re.search('[a-z0-9]*(inv[a-z0-9]*(\s)?gros)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['invoice gross'].values[i] = 0
        else:
            df_c['invoice gross'].values[i] = 1

        if (re.search('[a-z0-9]*((gros[a-z0-9]*(\s)?net))[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['gross net'].values[i] = 0
        else:
            df_c['gross net'].values[i] = 1

        if (re.search('[a-z0-9]*(amou[a-z0-9]*(\s)?disc)[a-z0-9]*', df_c['row_string_new'].values[i]) == None):
            df_c['amount discounts'].values[i] = 0
        else:
            df_c['amount discounts'].values[i] = 1

        '''if (re.search('[a-z]*((eff date)|(effdate)|(effdat)|(effective dat)|(tran dat))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['eff date'].values[i] = 0
        else:
            df_c['eff date'].values[i] = 1
        '''
        if (re.search('[a-z0-9]*((desc[a-z0-9]*(\s)?amou))[a-z0-9]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['description amount'].values[i] = 0
        else:
            df_c['description amount'].values[i] = 1

        if (re.search('[a-z0-9]*((disc[a-z0-9]*(\s)amou)|(amou[a-z0-9]*(\s)?disc))[a-z0-9]*',df_c['row_string_new'].values[i]) == None):
            df_c['discounts amount'].values[i] = 0
        else:
            df_c['discounts amount'].values[i] = 1

        if (re.search('[a-z0-9]*(gros)[a-z0-9 ]*', df_c['row_string_new'].values[i]) == None):
            df_c['gross'].values[i] = 0
        else:
            df_c['gross'].values[i] = 1

        if (re.search('[a-z0-9]*((inv[a-z0-9]*(\s)?loc)|(loc(\s)?inv))[a-z0-9]*',df_c['row_string_new'].values[i]) == None):
            df_c['invoice loc'].values[i] = 0
        else:
            df_c['invoice loc'].values[i] = 1

        if (re.search('[a-z0-9]*((memo[a-z0-9]*(\s)?inv)|(inv[a-z0-9]*(\s)?memo))[a-z0-9]*',
                          df_c['row_string_new'].values[i]) == None):
            df_c['memo invoice'].values[i] = 0
        else:
            df_c['memo invoice'].values[i] = 1

        if (re.search('[a-z0-9]*((num[a-z0-9]*(\s)?amou)|(amou[a-z0-9]*(\s)?num))[a-z0-9]*',
                          df_c['row_string_new'].values[i]) == None):
            df_c['number amount'].values[i] = 0
        else:
            df_c['number amount'].values[i] = 1

        if (re.search('[a-z0-9]*((disc[a-z0-9]*(\s)?net))[a-z0-9]*',
                          df_c['row_string_new'].values[i]) == None):
            df_c['discount net'].values[i] = 0
        else:
            df_c['discount net'].values[i] = 1

        if (re.search('[a-z0-9]*((date[a-z0-9]*(\s)?inv)|(inv)[a-z0-9]*(\s)?no|(inv)[a-z0-9]*(\s)?date)[a-z0-9]*',
                          df_c['row_string_new'].values[i]) == None):
            df_c['date invoice'].values[i] = 0
        else:
            df_c['date invoice'].values[i] = 1


        if (re.search('[a-z0-9]*((num[a-z0-9]*(\s)?dat)|(dat[a-z0-9]*(\s)?num))[a-z0-9]*',
                          df_c['row_string_new'].values[i]) == None):
            df_c['number date'].values[i] = 0
        else:
            df_c['number date'].values[i] = 1
    return df_c






###############################################################################
#feature_list::  vocab list, alpha_numeric_ratio, distance_from_top
#label::         is_heading
###############################################################################



#####################################
#vocab list
####################################
vocab=[]

vocab=[
'description gross',
'discount payment',
'amt balance',
'balance due',
'date type',
'due discount',
'invoice description',
'original amt',
#'reference original',
'type reference',
'date amount',
'amount deduction',
'invoice inv',
'gross discount',
'invoice gross',
'gross net',
'amount discounts',
'description amount',
'discounts amount',
'gross',
'invoice loc',
'memo invoice',
'number amount',
'date invoice',
'discount net',
'alpha_numeric_ratio',
'distance_from_top',
'number date'
]


label=df_c['is_heading'].reset_index()
#train_features,test_features,train_labels,test_labels=train_test_split(df_c,label,test_size=0.2,random_state=42)
#####################################################
#train data
#####################################################

df_c=clean_data(df_c)
df_c=func(df_c)
df_c['alpha_numeric_ratio']=df_c['row_string_new'].apply(alpha_func)
df_c['distance_from_top']=df_c['row_distanceFromTop'].apply(distance_from_top)
df_c=df_c.reset_index()

####################################################
#test data
####################################################

df_test=clean_data(df_test)
df_test=func(df_test)
df_test['alpha_numeric_ratio']=df_test['row_string_new'].apply(alpha_func)
df_test['distance_from_top']=df_test['row_distanceFromTop'].apply(distance_from_top)
df_test=df_test.reset_index()

####################################################
#combined train and test data for testing
####################################################

# test_features_combined=pd.concat([train_features,test_features]).reset_index()
# test_labels_combined=pd.concat([train_labels,test_labels]).reset_index()


####################################################
#modelling
####################################################

rf=RandomForestClassifier(n_estimators=40)
rf.fit(df_c[vocab],label)
pred=rf.predict(df_test[vocab])
df_temp=pd.DataFrame(pred,columns=['index','is_heading'])

df_test['is_heading']=df_temp['is_heading']

df_test.to_csv(test_file)


# print(classification_report(test_labels_combined['is_heading'],pred))
# print(confusion_matrix(test_labels_combined['is_heading'],pred))

