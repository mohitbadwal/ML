import pandas as pd
import random

import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from sklearn import model_selection
import string
import numpy as np
import sklearn
from sklearn import linear_model, datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dateutil.parser import parse

####################################################################################################

if len(sys.argv) == 3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
else:
    print("Format python filename train_dataset_path test_dataset_path")
    sys.exit(1)

data = pd.read_csv(train_file, sep=',', encoding='cp1256')
data2 = pd.read_csv(test_file, sep=',', encoding='cp1256', low_memory=False)

data = data[(data['page_type_final'] == 'remittance')]

data3 = data.append(data2, ignore_index=True)
data3['row_noOfCharacters'] = pd.cut(data3['row_noOfCharacters'], bins=10).cat.codes
data3 = data3.reset_index(drop=True)

data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)
data['row_noOfCharacters'] = data3['row_noOfCharacters'].loc[:data.shape[0] - 1].reset_index(drop=True)
data2['row_noOfCharacters'] = data3['row_noOfCharacters'].loc[data.shape[0]:data3.shape[0] - 1].reset_index(drop=True)
data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)

for i in data['check_checkNumber'].unique():
    for j in data[data['check_checkNumber'] == i]['page_pageNumber'].unique():
        temp = pd.DataFrame()
        temp = data[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j)]
        temp.reset_index(drop=True, inplace=True)
        first_row = 1
        last_row = temp.at[temp.shape[0] - 1, 'row_rowNumber']
        df = pd.DataFrame()
        df = temp[temp['is_total_final'] == 1]
        if df.empty:
            total_row_number = last_row
        else:
            total_row_number = df.reset_index(drop=True).at[df.shape[0] - 1, 'row_rowNumber']
        df2 = pd.DataFrame()
        df2 = temp[temp['is_heading'] == 1]
        if df2.empty:
            heading_row_number = first_row
        else:
            heading_row_number = df2.reset_index(drop=True).at[0, 'row_rowNumber']
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (
        data['row_rowNumber'] <= heading_row_number), 'remittance_result'] = 0
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (
        (data['row_rowNumber'] > heading_row_number) & (
        data['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (
        data['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0

for i in data2['check_checkNumber'].unique():
    for j in data2[data2['check_checkNumber'] == i]['page_pageNumber'].unique():
        temp = pd.DataFrame()
        temp = data2[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j)]
        temp.reset_index(drop=True, inplace=True)
        first_row = 1
        last_row = temp.at[temp.shape[0] - 1, 'row_rowNumber']
        df = pd.DataFrame()
        df = temp[temp['is_total'] == 1]
        if df.empty:
            total_row_number = last_row
        else:
            total_row_number = df.reset_index(drop=True).at[df.shape[0] - 1, 'row_rowNumber']
        df2 = pd.DataFrame()
        df2 = temp[temp['is_heading'] == 1]
        if df2.empty:
            heading_row_number = first_row
        else:
            heading_row_number = df2.reset_index(drop=True).at[0, 'row_rowNumber']
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (
        data2['row_rowNumber'] <= heading_row_number), 'remittance_result'] = 0
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (
        (data2['row_rowNumber'] > heading_row_number) & (
        data2['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (
        data2['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0

for i in range(0, data3.shape[0]):
    s = data3.at[i, 'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces

    total_charac = digits + letters + others
    data3.at[i, 'total_digits'] = digits / total_charac * 100
    data3.at[i, 'total_letters'] = letters / total_charac * 100
    data3.at[i, 'total_spaces'] = spaces / total_charac * 100
    data3.at[i, 'total_others'] = others / total_charac * 100

data3['total_digits_coded'] = pd.cut(data3['total_digits'], bins=10).cat.codes
data3['total_letters_coded'] = pd.cut(data3['total_letters'], bins=10).cat.codes
data3['total_spaces_coded'] = pd.cut(data3['total_spaces'], bins=10).cat.codes
data3['total_others_coded'] = pd.cut(data3['total_others'], bins=10).cat.codes
data3 = data3.reset_index(drop=True)

data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)
data['total_digits_coded'] = data3['total_digits_coded'].loc[:data.shape[0] - 1].reset_index(drop=True)
data2['total_digits_coded'] = data3['total_digits_coded'].loc[data.shape[0]:data3.shape[0] - 1].reset_index(drop=True)
data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)

data['total_letters_coded'] = data3['total_letters_coded'].loc[:data.shape[0] - 1].reset_index(drop=True)
data2['total_letters_coded'] = data3['total_letters_coded'].loc[data.shape[0]:data3.shape[0] - 1].reset_index(drop=True)
data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)

data['total_spaces_coded'] = data3['total_spaces_coded'].loc[:data.shape[0] - 1].reset_index(drop=True)
data2['total_spaces_coded'] = data3['total_spaces_coded'].loc[data.shape[0]:data3.shape[0] - 1].reset_index(drop=True)
data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)

data['total_others_coded'] = data3['total_others_coded'].loc[:data.shape[0] - 1].reset_index(drop=True)
data2['total_others_coded'] = data3['total_others_coded'].loc[data.shape[0]:data3.shape[0] - 1].reset_index(drop=True)
data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)

df3 = pd.DataFrame()
df3 = data2.reset_index(drop=True)

data['ratio_row_section'] = data['row_noOfCharacters'] / data['section_noOfCharacters']
df3['ratio_row_section'] = df3['row_noOfCharacters'] / df3['section_noOfCharacters']

data['amount_col_man'] = 0
for i in range(0, data.shape[0]):
    s = data.at[i, 'row_string']
    if '$' in s:
        data.at[i, 'amount_col_man'] = 1
    s = s.replace(',', '')
    s = s.replace('$', ' ')
    digits = re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s, flags=re.MULTILINE)
    for j in digits:
        if float(j) <= data.at[i, 'check_checkAmount']:
            data.at[i, 'amount_col_man'] = 1
            break

df3['amount_col_man'] = 0
for i in range(0, df3.shape[0]):
    s = df3.at[i, 'row_string']
    if '$' in s:
        df3.at[i, 'amount_col_man'] = 1
    s = s.replace(',', '')
    s = s.replace('$', ' ')
    digits = re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s, flags=re.MULTILINE)
    for j in digits:
        if float(j) <= df3.at[i, 'check_checkAmount']:
            df3.at[i, 'amount_col_man'] = 1
            break

pattern = re.compile(
    "Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))",
    re.IGNORECASE)


def dateFlag(x):
    global pattern
    if pattern.search(str(x)) is not None:
        return 1
    else:
        return 0


data['date_flag'] = data['row_string'].apply(dateFlag)
df3['date_flag'] = df3['row_string'].apply(dateFlag)


# data['date_amt_combined']=0
# df3['date_amt_combined']=0
# data.loc[(data['date_flag']==1) & (data['amount_col_man']==1),'date_amt_combined']=1
# data.loc[(data['date_flag']==0) & (data['amount_col_man']==1),'date_amt_combined']=1
# data.loc[(data['date_flag']==1) & (data['amount_col_man']==0),'date_amt_combined']=0
# data.loc[(data['date_flag']==0) & (data['amount_col_man']==0),'date_amt_combined']=0
#
# df3.loc[(df3['date_flag']==1) & (df3['amount_col_man']==1),'date_amt_combined']=1
# df3.loc[(df3['date_flag']==0) & (df3['amount_col_man']==1),'date_amt_combined']=1
# df3.loc[(df3['date_flag']==1) & (df3['amount_col_man']==0),'date_amt_combined']=0
# df3.loc[(df3['date_flag']==0) & (df3['amount_col_man']==0),'date_amt_combined']=0



def is_date(string):
    try:
        parse(string)
        return 1
    except ValueError:
        return 0


for i in range(0, df3.shape[0]):
    s = df3.at[i, 'row_string']
    words = s.split()
    for j in words:
        try:
            if is_date(j) == 1:
                df3.at[i, 'date_flag'] = 1
                break
        except OverflowError:
            continue

for i in range(0, data.shape[0]):
    s = data.at[i, 'row_string']
    words = s.split()
    for j in words:
        try:
            if is_date(j) == 1:
                data.at[i, 'date_flag'] = 1
                break
        except OverflowError:
            continue

# X=data[data['page_type_final']=='remittance'][['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# Y= data[data['page_type_final']=='remittance']['is_remittance_final'].reset_index(drop=True)

# X=pd.DataFrame()
# Y=pd.DataFrame()
# count=0
# for i in data['check_checkNumber'].unique():
#     X=data[data['']]

X_train = data[['date_flag', 'amount_col_man', 'ratio_row_section', 'row_noOfCharacters', 'remittance_result',
                'total_digits_coded', 'row_JosasisLRCoordinates_left', 'row_JosasisLRCoordinates_right',
                'row_distanceFromLeft', 'row_distanceFromTop']]
X_validation = df3[['date_flag', 'amount_col_man', 'ratio_row_section', 'row_noOfCharacters', 'remittance_result',
                    'total_digits_coded', 'row_JosasisLRCoordinates_left', 'row_JosasisLRCoordinates_right',
                    'row_distanceFromLeft', 'row_distanceFromTop']]
Y_train = data['is_remittance_final'].reset_index(drop=True)
Y_validation = df3['is_remittance_final_original'].reset_index(drop=True)

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

df3['is_remittance_mine']=predictions
# X_train=X_train[['ratio_row_section','total_digits_coded','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_col_man','date_flag','date_amt_combined','remittance_result','is_remittance_final','predictions','ocr_filepath']]
df3.to_csv(test_file)
#
#
#
# ###############################################################################################
#
#
