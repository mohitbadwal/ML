import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from sklearn import model_selection
import string
import numpy as np
import sklearn
from sklearn import linear_model, datasets,tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\new_result.csv", sep=',',encoding='cp1256')
print('initial shape = ',data.shape)

data = data[data['page_type_final'] == 'remittance']
print('final shape = ',data.shape)

data['row_noOfCharacters']=pd.cut(data['row_noOfCharacters'],bins=10).cat.codes

def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]

def cleanandstem(sentence):
    return cleaning(sentence)

for i in data['check_checkNumber'].unique():
    #total_pages=temp.at[0,'check_accountNumber']
    #print('i',i)
    for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
        #print('j', j)
        temp=pd.DataFrame()
        temp=data[(data['check_checkNumber']==i) & (data['page_pageNumber']==j)]
        temp.reset_index(drop=True,inplace=True)
        first_row=1
        last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
        df=pd.DataFrame()
        df=temp[temp['is_total_final']==1]
        if df.empty:
            total_row_number=last_row
        else:
            total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
        df2=pd.DataFrame()
        df2=temp[temp['is_heading']==1]
        if df2.empty:
            heading_row_number=first_row
        else:
            heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
        #print(first_row,last_row,heading_row_number,total_row_number)
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber']<= heading_row_number),'remittance_result']=0
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] > heading_row_number) & (data['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0


for i in range(0,data.shape[0]):
    s=data.at[i,'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces
    #alpha=letters
    #print(s,digits,letters,spaces,others)
    data.at[i,'total_digits'] = digits
    data.at[i, 'total_letters'] = letters
    data.at[i, 'total_spaces'] = spaces
    data.at[i, 'total_others'] = others
    #print(digits,':',alpha,'  ',data.at[i,'row_numberAlphaRatio'])


count=0
for i in data['check_checkNumber'].unique():
    for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
        count=0
        temp=pd.DataFrame()
        temp=data[(data['check_checkNumber']==i) & (data['page_pageNumber']==j)]
        temp.reset_index(drop=True,inplace=True)
        first_row=1
        last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
        df=pd.DataFrame()
        df=temp[temp['is_total_final']==1]
        if df.empty:
            total_row_number=last_row
        else:
            total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
        df2=pd.DataFrame()
        df2=temp[temp['is_heading']==1]
        if df2.empty:
            heading_row_number=first_row
        else:
            heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
        for k in range(total_row_number,heading_row_number-1,-1):
            data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] == k), 'distance_of_remittance_row'] = count
            count=count+1
        for k in range(first_row,heading_row_number):
            data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] == k), 'distance_of_remittance_row'] =
        for k in range(total_row_number+1,last_row+1):
            data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] == k), 'distance_of_remittance_row'] = 500

print(data['distance_of_remittance_row'])

vocab=['cnl','renb','date','flat','check','travel','newb','inst',
#'lpm',#'pckg'#'construction','bond'
]

tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english',vocabulary=vocab)
theString = tfidf.fit_transform(data['row_string'])
combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)

X=data['row_noOfCharacters']
X=pd.concat([X, combine1.reset_index(drop=True)], axis=1, ignore_index=True)
X = pd.concat([X, data['remittance_result'].reset_index(drop=True)], axis=1, ignore_index=True)
X = pd.concat([X, data['total_digits'].reset_index(drop=True)], axis=1, ignore_index=True)
X = pd.concat([X, data['total_letters'].reset_index(drop=True)], axis=1, ignore_index=True)
X = pd.concat([X, data['total_spaces'].reset_index(drop=True)], axis=1, ignore_index=True)
X = pd.concat([X, data['total_others'].reset_index(drop=True)], axis=1, ignore_index=True)
X = pd.concat([X, data['distance_of_remittance_row'].reset_index(drop=True)], axis=1, ignore_index=True)

Y = data.loc[:, 'is_remittance_final']
validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#Y_validation['pred_proba']=rfc.predict_proba(X_validation)
print (Y_train.shape[0],X_train.shape[0],X_validation.shape[0],Y_validation.shape[0])
#print(X_train)



