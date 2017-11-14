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

def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]

def cleanandstem(sentence):
    return cleaning(sentence)

#vocab=['policy','no','#','reference','pay','total','totals','ref','check','paid','transfer','trans','invoice','depo']
vocab=['account', 'agency', 'amt', 'approved',
       'balance', 'bond', 'bper', 'check', 'circle', 'cnl', 'code',
       'comm', 'commission', 'company', 'construction', 'corporation',
       'current', 'date', 'description', 'discount', 'effective',
       'eoc', 'fidelity', 'flat', 'gross', 'group', 'inst', 'insurance',
       'invoice', 'jun', 'llc', 'lpm', 'month', 'net', 'newb', 'north',
       'number', 'page', 'paid', 'payment', 'paysphere', 'pckg', 'policy',
       'premium', 'prf', 'renb', 'services', 'statement', 'total', 'totals',
       'trans', 'travel', 'type']
vocab=[
'cnl',
'renb',
'date',
'flat',
'check',
'travel',
'newb',
'inst',
'lpm',
'pckg',
'construction','bond'
]

tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english',vocabulary=vocab)
theString = tfidf.fit_transform(data['row_string'])
combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)
X=combine1.reset_index(drop=True)
#X = pd.concat([combine1.reset_index(drop=True), data['page_noOfRows'].reset_index(drop=True)], axis=1, ignore_index=True)
#X = pd.concat([X, data['page_noOfSections'].reset_index(drop=True)], axis=1, ignore_index=True)
Y = data.loc[:, 'is_remittance_final']
validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


importances = rfc.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# # Print the feature ranking
print("Feature ranking:")
#print(train_features.columns)
myList=list()
for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, vocab[indices[f]], importances[indices[f]]))
    myList.append(vocab[indices[f]])

# # Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), myList)
plt.xlim([-1, X_train.shape[1]])
plt.show()


count=0
for i in data['check_checkNumber'].unique():
    #total_pages=temp.at[0,'check_accountNumber']
    print('i',i)
    for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
        print('j', j)
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
        print(first_row,last_row,heading_row_number,total_row_number)
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber']<= heading_row_number),'remittance_result']=0
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] > heading_row_number) & (data['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0


data.to_csv('new_result.csv')

#X=data['remittance_result'].reset_index(drop=True)
#X.values.reshape(-1, 1)
#X = pd.concat([combine1.reset_index(drop=True), data['remittance_result'].reset_index(drop=True)], axis=1, ignore_index=True)
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
#rfc = RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

a=data[data['is_remittance_final']==data['remittance_result']].shape[0]
b=data[data['is_remittance_final']!=data['remittance_result']].shape[0]
c=data[(data['is_remittance_final']==data['remittance_result']) & (data['is_remittance_final']==1)].shape[0]
d=data[(data['is_remittance_final']==data['remittance_result']) & (data['is_remittance_final']==0)].shape[0]
print(a,b,c,d)
print('total=',data.shape[0])