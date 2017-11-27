import sys
from sklearn.learning_curve import learning_curve
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
from sklearn.multiclass import OneVsRestClassifier

if len(sys.argv) == 3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
else:
    print("Format python filename train_dataset_path test_dataset_path")
    sys.exit(1)

data = pd.read_csv(train_file, sep=',', encoding='latin-1')
data_test = pd.read_csv(test_file, sep=',', encoding='latin-1')
data_test['is_remittance_final_original'] = data_test['is_remittance_final']
data_test['is_remittance_final'] = data_test.groupby(['check_checkNumber', 'page_pageNumber'])[
    'is_remittance_final'].transform('max')
p = data_test.groupby(['check_checkAmount', 'check_checkNumber', 'page_pageNumber', 'is_remittance_final']).apply(
    lambda x: x.groupby(['page_pageType'])['row_string'].apply(lambda x: ''.join(x)))

k = data.groupby(['check_checkAmount', 'check_checkNumber', 'page_pageNumber']).apply(
    lambda x: x.groupby(['page_type_final'])['row_string'].apply(lambda x: ''.join(x)))
df_c = pd.Series.to_frame(k).reset_index()

df_c_test = pd.Series.to_frame(p).reset_index()

df_c['page_type'] = df_c['page_type_final']
df_c['page_type_final'] = df_c['page_type_final'].replace(
    ['remittance advice', 'invoice', 'invoice template', 'reminder', 'memo', 'accidental death and dismemberment plan',
     'improper remittance', 'template page', 'bill', 'cheque voucher template', 'check voucher'], 'others')
dic = {'check': 0, 'envelope': 1, 'remittance': 2, 'others': 3}
df_c['page_type_final'] = df_c['page_type_final'].map(dic)

df_c['amount_match'] = None
df_c_test['amount_match'] = None


def text_clean(df_c):
    df_c['row_string_new'] = df_c['row_string'].apply(lambda x: x.lower())
    # ==============================================================================
    #     df_c['row_string_new']=df_c['row_string'].str.replace('[^\w\s]',' ')
    #
    #
    #     stop=set(stopwords.words('english'))
    #
    #     df_c['row_string_new']=df_c['row_string_new'].apply(lambda x: [item for item in x.split()  if item not in stop])
    #
    #     df_c['row_string_new']=df_c['row_string_new'].apply(lambda x: ' '.join([i for i in x]))
    return df_c


#
# ==============================================================================
#####################################################################################################



def envelope(df_c):
    list_r = []
    for i in df_c['row_string_new']:
        if re.search('[1i]{10,}', i):
            list_r.append(1)
        else:
            list_r.append(0)

    df_c['reg_f'] = list_r

    list_g = []
    for i in df_c['row_string_new']:
        if re.search('([li\,\.\'\"\?\!\(\\)rndtph1j]{10,})', i):
            list_g.append(len(tuple(re.finditer('([li\,\.\'\"\?\!\(\\)rndtph1j]{10,})', i))))
        else:
            list_g.append(0)

    df_c['reg_g'] = list_g

    list_m = []
    for i in df_c['row_string_new']:
        if re.search('([m][a][i][l][e][d]) | ([ ][m][a][i][l][e][d][ ])|([ ][m][a][i][l][e][d])', i):
            list_m.append(
                len(tuple(re.finditer('([m][a][i][l][e][d]) | ([ ][m][a][i][l][e][d][ ])|([ ][m][a][i][l][e][d])', i))))
        else:
            list_m.append(0)

    df_c['reg_m'] = list_m

    list_s = []
    for i in df_c['row_string_new']:
        if re.search(
                '([s][h][i][p][m][e][n][t])|([s][h][i][p][p][i][n][g])|([s][h][i][p][ ][d][a][t])|([s][h][i][p][ ][t][o])|([s][h][i][p][d][a][t])|([s][h][i][p][t][o])',
                i):
            list_s.append(len(tuple(re.finditer(
                '([s][h][i][p][m][e][n][t])|([s][h][i][p][p][i][n][g])|([s][h][i][p][ ][d][a][t])|([s][h][i][p][ ][t][o])|([s][h][i][p][d][a][t])|([s][h][i][p][t][o])',
                i))))
        else:
            list_s.append(0)

    df_c['reg_s'] = list_s

    list_ret = []
    for i in df_c['row_string_new']:
        if re.search(
                '([r][e][t][u][r][n][ ][s][e][r][v][i][c][e])|([r][e][t][u][r][n][s][e][r][v][i][c][e])|([r][e][t][u][r][n][t][o])|([r][e][t][u][r][n][ ][t][o])',
                i):
            list_ret.append(len(tuple(re.finditer(
                '([r][e][t][u][r][n][ ][s][e][r][v][i][c][e])|([r][e][t][u][r][n][s][e][r][v][i][c][e])|([r][e][t][u][r][n][t][o])|([r][e][t][u][r][n][ ][t][o])',
                i))))
        else:
            list_ret.append(0)

    df_c['reg_ret'] = list_ret

    list_e = []
    for i in df_c['row_string_new']:
        if re.search('(([e][n][v][e][l][o][p][e]))', i):
            list_e.append(len(tuple(re.finditer('(([e][n][v][e][l][o][p][e]))', i))))
        else:
            list_e.append(0)

    df_c['reg_e'] = list_e

    list_z = []
    for i in df_c['row_string_new']:
        if re.search('(([z][i][p]))', i):
            list_z.append(len(tuple(re.finditer('(([z][i][p]))', i))))
        else:
            list_z.append(0)

    df_c['reg_z'] = list_z

    list_a = []
    for i in df_c['row_string_new']:
        if re.search('(([a][d][d][r][e][s][s]))', i):
            list_a.append(len(tuple(re.finditer('(([a][d][d][r][e][s][s]))', i))))
        else:
            list_a.append(0)

    df_c['reg_a'] = list_a

    list_p = []
    for i in df_c['row_string_new']:
        if re.search('(([p][o][s][t][ ])|([p][o][s][t][a][g][e]))', i):
            list_p.append(len(tuple(re.finditer('(([p][o][s][t][ ])|([p][o][s][t][a][g][e]))', i))))
        else:
            list_p.append(0)

    df_c['reg_p'] = list_p

    list_fe = []
    for i in df_c['row_string_new']:
        if re.search(
                '(([p][i][t][n][e][y])|([u][p][s])|([n][e][o][p])|([h][a][s][l][e][r])|([f][i][r][s][t][ ][c][l][a][s][s]))',
                i):
            list_fe.append(len(tuple(re.finditer(
                '(([p][i][t][n][e][y])|([u][p][s])|([n][e][o][p])|([h][a][s][l][e][r])|([f][i][r][s][t][ ][c][l][a][s][s]))',
                i))))
        else:
            list_fe.append(0)

    df_c['reg_fe'] = list_fe

    return df_c


def checks(df_c):
    list_chase = []
    for i in df_c['row_string_new']:
        if re.search('([a-zA-Z0-9]*(chase))|([a-z0-9]*(american express bank))|([a-z0-9]*(goldman sachs bank))', i):
            list_chase.append(1)
        else:
            list_chase.append(0)

    df_c['reg_chase'] = list_chase

    list_da = []
    for i in df_c['row_string_new']:
        if re.search('([a-z0-9]*(pay )[a-z0-9]*)', i):
            list_da.append(1)
        else:
            list_da.append(0)

    df_c['reg_da'] = list_da

    list_digits = []
    for i in df_c['row_string_new']:
        if re.search(
                '([ ][o][n][e])|([t][w][o])|([t][h][r][e][e])|([f][o][u][r])|([f][i][v][e])|([s][i][x])|([s][e][v][e][n])|([ ][e][i][g][h][t])|([n][i][n][e])|([ ][t][e][n][ ])|([t][w][e][n])|([t][h][i][r][t])|([f][o][r][t])|([f][i][f][t])|([e][i][g][h][t][y])|([h][u][n][d])|([o][u][s][a][n])',
                i):
            list_digits.append(1)
        else:
            list_digits.append(0)

    df_c['reg_d'] = list_digits

    list_order = []
    for i in df_c['row_string_new']:
        if re.search('[^(re)](order)', i):
            list_order.append(1)
        else:
            list_order.append(0)

    df_c['reg_order'] = list_order

    list_currency = []
    for i in df_c['row_string_new']:
        if re.search('([a-zA-Z0-9]*(cents))|((one)(cent))|((one )(cent))|([a-zA-Z]*(dollar))', i):
            list_currency.append(1)
        else:
            list_currency.append(0)

    df_c['reg_currency'] = list_currency

    list_auth = []
    for i in df_c['row_string_new']:
        if re.search(
                '([a-zA-Z0-9]*(signature))|([a-zA-Z0-9]*(authorized))|([a-zA-Z0-9]*(watermark))|([a-zA-Z0-9]*(security))',
                i):
            list_auth.append(1)
        else:
            list_auth.append(0)

    df_c['reg_auth'] = list_auth

    list_col = []
    for i in df_c['row_string_new']:
        if re.search('([a-zA-Z0-9]*(background))|([a-zA-Z0-9]*(colored))|([a-zA-Z0-9]*(void))|([a-zA-Z0-9]*(contains))',
                     i):
            list_col.append(1)
        else:
            list_col.append(0)

    df_c['reg_col'] = list_col
    return df_c


def amount_match_func(df_c):
    d = ''
    l1 = ''
    l2 = ''
    k = ''
    for i in range(0, df_c.shape[0]):
        d = ("{:,f}".format(df_c['check_checkAmount'].values[i])).replace('0000', '')
        k = ("{:f}".format(df_c['check_checkAmount'].values[i]))
        l1 = k.split('.')[0]
        l2 = k.split('.')[1]
        l2 = l2.replace('0000', '')
        l1 = l1 + '.' + l2
        if ((d in df_c['row_string_new'].values[i]) | (l1 in df_c['row_string_new'].values[i]) | (
            str(df_c['check_checkAmount'].values[i]) in df_c['row_string_new'])):
            df_c['amount_match'].values[i] = 1
        else:
            df_c['amount_match'].values[i] = 0
        d = ''
        k = ''
        l1 = ''
        l2 = ''

    for i in range(0, df_c.shape[0]):
        if (df_c['page_pageNumber'].values[i] == 1):
            df_c['amount_match'].values[i] = 0

    return df_c


def func(df_c):
    df_c['totals'] = None
    # df_c['discount']=None
    df_c['net am'] = None
    df_c['net pa'] = None
    df_c['net to'] = None
    df_c['gross am'] = None
    df_c['gross ag'] = None
    df_c['amount pa'] = None
    df_c['paid'] = None
    # df_c['paid pay']=None
    df_c['inv am'] = None
    df_c['inv n'] = None
    df_c['inv date'] = None
    df_c['policy am'] = None
    df_c['policy writ'] = None
    df_c['policy assured'] = None
    df_c['grand tot'] = None
    df_c['name'] = None
    df_c['balance'] = None
    df_c['item'] = None
    df_c['type'] = None
    df_c['disc'] = None
    df_c['eff date'] = None
    df_c['vendor'] = None
    df_c['debit credit'] = None
    for i in range(0, df_c.shape[0]):
        if (re.search('[a-z]*(totals)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['totals'].values[i] = 0
        else:
            df_c['totals'].values[i] = 1

        if (re.search('[a-z]*((net am)|(netam))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['net am'].values[i] = 0
        else:
            df_c['net am'].values[i] = 1

        if (re.search('[a-z]*((net pa)|(netpa))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['net pa'].values[i] = 0
        else:
            df_c['net pa'].values[i] = 1

        if (re.search('[a-z]*((net to)|(netto))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['net to'].values[i] = 0
        else:
            df_c['net to'].values[i] = 1

        if (re.search('[a-z]*((gross am)|(grossam))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['gross am'].values[i] = 0
        else:
            df_c['gross am'].values[i] = 1

        if (re.search('[a-z]*((gross ag)|(grossag))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['gross ag'].values[i] = 0
        else:
            df_c['gross ag'].values[i] = 1

        if (re.search('[a-z]*((amount pa)|(amt pa)|(amountpa)|(amtpa))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['amount pa'].values[i] = 0
        else:
            df_c['amount pa'].values[i] = 1

        if (re.search('[a-z]*(paid)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['paid'].values[i] = 0
        else:
            df_c['paid'].values[i] = 1

        if (re.search('[a-z]*((inv am)|(invoice am)|(invam)|(invoiceam))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['inv am'].values[i] = 0
        else:
            df_c['inv am'].values[i] = 1

        if (re.search('[a-z]*((invoice n)|(inv n)|(invn)|(invoicen))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['inv n'].values[i] = 0
        else:
            df_c['inv n'].values[i] = 1

        if (re.search('[a-z]*((invoice date)|(inv date)|(invdate)|(invoicedate))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['inv date'].values[i] = 0
        else:
            df_c['inv date'].values[i] = 1

        if (re.search('[a-z]*((policy am)|(policyam))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['policy am'].values[i] = 0
        else:
            df_c['policy am'].values[i] = 1

        if (re.search('[a-z]*((policy writ)|(policywrit))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['policy writ'].values[i] = 0
        else:
            df_c['policy writ'].values[i] = 1

        if (re.search('[a-z]*((policy assured)|(assured policy)|(policyassured)|(assuredpolicy))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['policy assured'].values[i] = 0
        else:
            df_c['policy assured'].values[i] = 1

        if (re.search('[a-z]*((grand tot)|(grandtot))[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['grand tot'].values[i] = 0
        else:
            df_c['grand tot'].values[i] = 1

        if (re.search('[a-z]*(name)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['name'].values[i] = 0
        else:
            df_c['name'].values[i] = 1

        if (re.search('[a-z]*(balance)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['balance'].values[i] = 0
        else:
            df_c['balance'].values[i] = 1

        if (re.search('[a-z]*(item)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['item'].values[i] = 0
        else:
            df_c['item'].values[i] = 1

        if (re.search('[a-z]*(type)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['type'].values[i] = 0
        else:
            df_c['type'].values[i] = 1

        if (re.search('[a-z]*(disc)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['disc'].values[i] = 0
        else:
            df_c['disc'].values[i] = 1

        if (re.search('[a-z]*((eff da)|(effda)|(effective da)|(effectiveda))[a-z]*',
                      df_c['row_string_new'].values[i]) == None):
            df_c['eff date'].values[i] = 0
        else:
            df_c['eff date'].values[i] = 1

        if (re.search('[a-z]*(vend)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['vendor'].values[i] = 0
        else:
            df_c['vendor'].values[i] = 1

        if (re.search('[a-z]*(debit credit)[a-z]*', df_c['row_string_new'].values[i]) == None):
            df_c['debit credit'].values[i] = 0
        else:
            df_c['debit credit'].values[i] = 1

    return df_c


label = df_c['page_type_final']

vocab2 = ['totals',
          'disc',
          'net am',
          'vendor',
          'debit credit',

          'amount pa',
          'paid',
          'eff date',

          'inv n',
          'inv date',

          'policy assured',
          'grand tot',
          'name', 'balance', 'type', 'item',

          'reg_m', 'reg_s', 'reg_ret', 'reg_e', 'reg_z', 'reg_a', 'reg_p', 'reg_fe',

          'reg_chase', 'reg_da', 'reg_d', 'reg_order', 'reg_currency', 'reg_auth', 'reg_col', 'amount_match',
          'amount_count', 'date', 'check_identifier'
          ]

final_train_data = text_clean(df_c)
final_train_data = func(df_c)
final_train_data = envelope(df_c)
final_train_data = checks(df_c)
final_train_data = amount_match_func(df_c)
final_train_data['amount_count'] = final_train_data['row_string_new'].apply(
    lambda x: len(tuple(re.finditer('[0-9\,]+(\.)(0|1|2|3|4|5|6|7|8|9)(0|1|2|3|4|5|6|7|8|9)[a-z\$ ]', x))) if re.search(
        '[0-9\,]+(\.)(0|1|2|3|4|5|6|7|8|9)(0|1|2|3|4|5|6|7|8|9)[a-z\$ ]', x) else 0)
final_train_data['date'] = final_train_data['row_string_new'].apply(lambda x: len(tuple(re.finditer(
    '((([0-3]?[0-9][/][0-1]?[0-9][/](((20)[0-9]{2})|([1-2]{1}[1-5]{1})))|([0-1]?[0-9][/][0-3]?[0-9][/](((20)[0-9]{2})|([1-2]{1}[1-5]{1})))|([0-3]?[1-9][ ]?(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?))))',
    x))) if re.search(
    '(([0-3]?[0-9][/][0-1]?[0-9][/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][-/][0-3]?[0-9][/-](([0-9]{4})|([0-9]{2})))|([0-3]?[1-9][ ]?(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)))',
    x) else 0)
final_train_data['check_identifier'] = final_train_data['page_pageNumber'].apply(lambda x: 1 if x == 1 else 0)

final_train_data = final_train_data.reset_index()

final_test_data = text_clean(df_c_test)
final_test_data = func(df_c_test)
final_test_data = envelope(df_c_test)
final_test_data = checks(df_c_test)
final_test_data = amount_match_func(df_c_test)
final_test_data['amount_count'] = final_test_data['row_string_new'].apply(
    lambda x: len(tuple(re.finditer('[0-9\,]+(\.)(0|1|2|3|4|5|6|7|8|9)(0|1|2|3|4|5|6|7|8|9)[a-z\$ ]', x))) if re.search(
        '[0-9\,]+(\.)(0|1|2|3|4|5|6|7|8|9)(0|1|2|3|4|5|6|7|8|9)[a-z\$ ]', x) else 0)
final_test_data['date'] = final_test_data['row_string_new'].apply(lambda x: len(tuple(re.finditer(
    '((([0-3]?[0-9][/][0-1]?[0-9][/](((20)[0-9]{2})|([1-2]{1}[1-5]{1})))|([0-1]?[0-9][/][0-3]?[0-9][/](((20)[0-9]{2})|([1-2]{1}[1-5]{1})))|([0-3]?[1-9][ ]?(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?))))',
    x))) if re.search(
    '(([0-3]?[0-9][/][0-1]?[0-9][/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][-/][0-3]?[0-9][/-](([0-9]{4})|([0-9]{2})))|([0-3]?[1-9][ ]?(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)))',
    x) else 0)
final_test_data['check_identifier'] = final_test_data['page_pageNumber'].apply(lambda x: 1 if x == 1 else 0)

final_test_data = final_test_data.reset_index()

train_labels = label.reset_index()

ovr = OneVsRestClassifier(RandomForestClassifier(n_estimators=300))

ovr.fit(final_train_data[vocab2], train_labels['page_type_final'])
pred = ovr.predict(final_test_data[vocab2])

x = pd.DataFrame()
x['pred'] = pred
data_temp = pd.DataFrame()

data_temp['pred'] = x['pred']
data_temp['page_pageNumber'] = df_c_test['page_pageNumber']
data_temp['check_checkNumber'] = df_c_test['check_checkNumber']

data_test['page_type'] = None
for j in range(0, data_temp.shape[0]):
    a = data_temp.at[j, 'check_checkNumber']
    b = data_temp.at[j, 'page_pageNumber']
    data_test.loc[(data_test['check_checkNumber'] == a) & (data_test['page_pageNumber'] == b), 'pred'] = data_temp.at[
        j, 'pred']

# path='C:\\Users\\gaurav.subedar\\Desktop\\doc_classification'
data_test.to_csv(test_file)





