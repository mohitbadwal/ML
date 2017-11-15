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
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import os
import seaborn as sb
import string
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier

dataset = pd.read_csv(r'D:\backup\PycharmProjects\test\Image '
                      r'Batches-20171017T131547Z-001\Not_Success_rows_ver_clean.csv',
                      encoding='cp1256')
dataset_test = pd.read_csv(r'D:\backup\PycharmProjects\test\Image Batches-20171017T131547Z-001\python_files'
                           r'\test.csv',
                           encoding='cp1256')
dataset['sub1'] = dataset.check_noOfPages - dataset.page_pageNumber
dataset.sub1 = dataset.sub1.apply(lambda x: 0 if x < 0 else x)


def getAlphaNumRatio(x):
    digits = sum(c.isdigit() for c in x)
    letters = sum(c.isalpha() for c in x)
    if letters == 0:
        return 0
    else:
        # return float(float(digits)/float(letters))
        if (digits / letters) < 0.1:
            return 1
        else:
            return 0


def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    # punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    # l = [word for word in l if len(word) > 2]
    return " ".join(l)


def func(df_c):
    df_c['discount'] = None
    df_c['net am'] = None
    df_c['net pa'] = None
    df_c['net to'] = None
    df_c['net pr'] = None
    df_c['gross am'] = None
    df_c['Flat'] = None
    df_c['policy'] = None
    df_c['gross ag'] = None
    df_c['amount pa'] = None
    df_c['numb date'] = None
    df_c['paid'] = None
    df_c['paid pay'] = None
    df_c['pol writ'] = None
    df_c['inv am'] = None
    df_c['inv n'] = None
    df_c['inv date'] = None
    df_c['policy am'] = None
    df_c['policy writ'] = None
    df_c['pol am'] = None
    df_c['pol'] = None
    df_c['policy assured'] = None
    # df_c['grand tot'] = None
    df_c['name'] = None
    df_c['balance'] = None
    df_c['item'] = None
    df_c['type'] = None
    df_c['disc'] = None
    df_c['eff date'] = None
    df_c['vendor'] = None
    df_c['debit credit'] = None
    df_c['commission'] = None
    df_c['effective'] = None
    df_c['description'] = None
    for i in range(0, df_c.shape[0]):

        # if re.search('[a-z]*((description)|(desc))[a-z]*', df_c['row_string_new'].values[i]) == None:
        #    df_c['description'].values[i] = 0
        # else:
        #    df_c['description'].values[i] = 1

        if (re.search('[a-z]*((discount)|(disc )|(description)|(desc ))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['discount'].values[i] = 0
        else:
            df_c['discount'].values[i] = 1

        if (re.search(
                '[a-z]*((net am)|(netam)|(net(\s)?check(\s)?am)|(amt(\s)?net)|(amount(\s)?net)|(inv am)|(invoice am)|(invam)|(invoiceam))[a-z]*',
                df_c['row_string_new'].values[i]) == None):
            df_c['net am'].values[i] = 0
        else:
            df_c['net am'].values[i] = 1

        if (re.search('[a-z]*((net pa)|(netpa))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['net pa'].values[i] = 0
        else:
            df_c['net pa'].values[i] = 1

        if (re.search('[a-z]*((premium(\s)?net)|(net(\s)?pr)|(pr(\s)?net)|(net pa)|(netpa))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['net pr'].values[i] = 0
        else:
            df_c['net pr'].values[i] = 1

        if (re.search('[a-z]*((net to)|(netto))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['net to'].values[i] = 0
        else:
            df_c['net to'].values[i] = 1

        if (re.search('[a-z]*((gross am)|(grossam)|(gross ag)|(grossag)|(agcy(\s)?gross)|(agency(\s)?gross))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['gross am'].values[i] = 0
        else:
            df_c['gross am'].values[i] = 1

        if (re.search('[a-z]*((gross ag)|(grossag)|(agcy gross)|(agency))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['gross ag'].values[i] = 0
        else:
            df_c['gross ag'].values[i] = 1

        if (re.search('[a-z]*((amount pa)|(amt pa)|(amountpa)|(amtpa)|(previously(\s)?paid))[a-z]*',
                      df_c['row_string_new'].values[i]) is None):
            df_c['amount pa'].values[i] = 0
        else:
            df_c['amount pa'].values[i] = 1

        # ==============================================================================
        #     if(re.search('[a-z]*((paid pay)|(paidpay))[a-z]*',df_c['row_string_new'].values[i])==None):
        #         df_c['paid pay'].values[i]=0
        #     else:
        #         df_c['paid pay'].values[i]=1
        # ==============================================================================
        if (re.search('[a-z]*((inv am)|(invoice am)|(invam)|(invoiceam))[a-z]*',
                      df_c['row_string_new'].values[i]) is None):
            df_c['inv am'].values[i] = 0
        else:
            df_c['inv am'].values[i] = 1

        if (re.search(
                '[a-z]*((inv no)|(invoice no)|(invoice numb)|(inv numb)|(invno)|(invoiceno)|(invoicenumb)|(invnumb)|(item(\s)?n))[a-z]*',
                df_c['row_string_new'].values[i]) == None):
            df_c['inv n'].values[i] = 0
        else:
            df_c['inv n'].values[i] = 1

        if (re.search('[a-z]*((inv dat)|(invdat)|(invoice dat)|(invoicedat))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['inv date'].values[i] = 0
        else:
            df_c['inv date'].values[i] = 1

        if (re.search('[a-z]*((policy am)|(policyam)|(pol am)|(polam))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['policy am'].values[i] = 0
        else:
            df_c['policy am'].values[i] = 1

        if (re.search('[a-z]*((policy writing)|(policy writ)|(policywrit)|(pol writ))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['policy writ'].values[i] = 0
        else:
            df_c['policy writ'].values[i] = 1

        if (re.search(
                '[a-z]*((policy(\s)?number)|(policy(\s)?no)|(policy(\s)?#)|(pol(\s)?no)|(pol(\s)?#)|(pol(\s)?type)|(policy(\s)?type)|(pol(\s)?number)|(policy(\s)?writ)|(pol(\s)?writ))[a-z]*',
                df_c['row_string_new'].values[i]) == None):
            df_c['policy'].values[i] = 0
        else:
            df_c['policy'].values[i] = 1

        if (re.search('[a-z]*(name)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['name'].values[i] = 0
        else:
            df_c['name'].values[i] = 1

        if (re.search('[a-z]*(balance)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['balance'].values[i] = 0
        else:
            df_c['balance'].values[i] = 1

        if (re.search('[a-z]*((item)|(item number))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['item'].values[i] = 0
        else:
            df_c['item'].values[i] = 1

        if (re.search('[a-z]*(type)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['type'].values[i] = 0
        else:
            df_c['type'].values[i] = 1

        '''if (re.search('[a-z]*((eff date)|(effdate)|(effdat)|(effective dat)|(tran dat))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['eff date'].values[i] = 0
        else:
            df_c['eff date'].values[i] = 1
        '''
        if (re.search('[a-z]*((number dat)|(no dat)|(account dat)|(acc dat))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['numb date'].values[i] = 0
        else:
            df_c['numb date'].values[i] = 1

        if (re.search('[a-z]*((Flat)|(flat))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['Flat'].values[i] = 0
        else:
            df_c['Flat'].values[i] = 1

        if (re.search(
                '[a-z]*((comm(\s)?%)|(com(\s)?%)|(commission(\s)?rate)|(comm(\s)?rate)|(agency(\s)?comm)|(agcy(\s)?com))[a-z]*',
                df_c['row_string_new'].values[i]) == None):
            df_c['commission'].values[i] = 0
        else:
            df_c['commission'].values[i] = 1

        if (re.search('[a-z]*((eff )|(effective)|(eff date)|(effdate)|(effdat)|(effective dat)|(tran dat))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['effective'].values[i] = 0
        else:
            df_c['effective'].values[i] = 1

            # if (re.search('[a-z]*(vend)[a-z]*', df_c['row_string_new'].values[i]) == None):
            #    df_c['vendor'].values[i] = 0
            # else:
            #    df_c['vendor'].values[i] = 1

            # if re.search('[a-z]*(debit credit)[a-z]*', df_c['row_string_new'].values[i]) == None:
            #    df_c['debit credit'].values[i] = 0
            # else:
            #    df_c['debit credit'].values[i] = 1

    return df_c


dataset['row_string_new'] = dataset['row_string'].apply(cleaning)
df_new = func(dataset)

dataset_test['row_string_new'] = dataset_test['row_string'].apply(cleaning)
df_new_test = func(dataset_test)

vocab2 = ['discount',
          'net am',
          # 'net pa',
          'net pr',
          # 'net to',
          # 'vendor',
          # 'debit credit',
          # 'description',
          'amount pa',
          'policy',
          # 'policy am',
          # 'policy writ',
          # 'eff date',
          'inv n',
          # 'inv am',
          'inv date',
          'name', 'type',
          # 'item',
          # 'balance',
          'numb date',
          # 'commission',
          'effective',  # 'gross ag',
          'gross am',
          'row_numberAlphaRatio',
          'row_distanceFromTop',
          'row_rowNumber',
          # 'sub1'
          ]
label = df_new['is_heading']
df_new['row_numberAlphaRatio'] = df_new['row_string_new'].apply(getAlphaNumRatio)
df_new_test['row_numberAlphaRatio'] = df_new_test['row_string_new'].apply(getAlphaNumRatio)

rf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10)
# rf =MLPClassifier()
# rf=SGDClassifier(penalty="l2",alpha=0.0001)
# rf = MultinomialNB(alpha=0.01)
rf.fit(df_new[vocab2], label)
pred = rf.predict(df_new_test[vocab2])
er = pd.DataFrame({"is_heading": pred})
dataset_test = pd.concat([dataset_test, er], axis=1)
dataset_test.to_csv("test.csv")
# pred = rf.predict(df_new[vocab2])
