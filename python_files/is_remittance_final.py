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

data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Not_Success_rows_ver.csv", sep=',',encoding='cp1256')
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
    print('i',i)
    for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
        print('j',j)
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

data[['check_checkNumber','page_pageNumber','row_rowNumber','is_total_final','is_heading','is_remittance_final','remittance_result']].to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\new_result_by_badwal.csv")

data=data.reset_index(drop=True)

for i in range(0,data.shape[0]):
    s=data.at[i,'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces
    #alpha=letters
    #print(s,digits,letters,spaces,others)
    total_charac=digits+letters+spaces+others
    data.at[i,'total_digits'] = digits/total_charac
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
        count=total_row_number-last_row
        for k in range(last_row,first_row-1,-1):
            data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] == k), 'distance_of_remittance_row'] = count
            count=count+1


#print(data['distance_of_remittance_row'])

vocab=['cnl','renb','date','flat','check','travel','newb','inst',
#'lpm',#'pckg'#'construction','bond'
]

# tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english',vocabulary=vocab)
# theString = tfidf.fit_transform(data['row_string'])
# combine1 = pd.DataFrame(theString.todense())
# combine1.columns = tfidf.get_feature_names()
# print(combine1.columns)

validation_size=0.3
X = data[['check_checkNumber','page_pageNumber','row_rowNumber','row_string','is_total_final','is_heading','hyper_link1','ocr_filepath','row_noOfCharacters','remittance_result','total_digits','total_letters','total_spaces','total_others','distance_of_remittance_row']]
Y = data['is_remittance_final']
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
print(X_train.shape,Y_train.shape,X_validation.shape,Y_validation.shape)

tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english',vocabulary=vocab)
theString = tfidf.fit_transform(X_train['row_string'])
theString2 = tfidf.transform(X_validation['row_string'])
combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)
combine2 = pd.DataFrame(theString2.todense())
combine2.columns = tfidf.get_feature_names()
print(combine2.columns)

X_train=X_train.reset_index(drop=True)
Y_train=Y_train.reset_index(drop=True)
X_validation=X_validation.reset_index(drop=True)
Y_validation=Y_validation.reset_index(drop=True)
X_train=X_train[['row_noOfCharacters','remittance_result','total_digits','total_letters','total_spaces','total_others']]


X_validation1=X_validation
X_validation=X_validation[['row_noOfCharacters','remittance_result','total_digits','total_letters','total_spaces','total_others']]





X_train = pd.concat([X_train, combine1.reset_index(drop=True)], axis=1, ignore_index=True)
X_validation = pd.concat([X_validation, combine2.reset_index(drop=True)], axis=1, ignore_index=True)

# X=data['row_noOfCharacters']
# print(X.head(2))
# X=pd.concat([X, combine1.reset_index(drop=True)], axis=1, ignore_index=True)
# X = pd.concat([X, data['remittance_result'].reset_index(drop=True)], axis=1, ignore_index=True)
# X = pd.concat([X, data['total_digits'].reset_index(drop=True)], axis=1, ignore_index=True)
# X = pd.concat([X, data['total_letters'].reset_index(drop=True)], axis=1, ignore_index=True)
# X = pd.concat([X, data['total_spaces'].reset_index(drop=True)], axis=1, ignore_index=True)
# X = pd.concat([X, data['total_others'].reset_index(drop=True)], axis=1, ignore_index=True)
# X = pd.concat([X, data['distance_of_remittance_row'].reset_index(drop=True)], axis=1, ignore_index=True)
# print(X.head(2))
# Y = data.loc[:, 'is_remittance_final']
# validation_size = 0.3
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
#
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#
# #Y_validation['pred_proba']=rfc.predict_proba(X_validation)
# print (Y_train.shape[0],X_train.shape[0],X_validation.shape[0],Y_validation.shape[0])
# #print(X_train)
#
#

# importances = rfc.feature_importances_
# print(importances)
# std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
#
# # # Print the feature ranking
# print("Feature ranking:")
# #print(train_features.columns)
# for f in range(X_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
# # # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X_train.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X_train.shape[1]), indices)
# plt.xlim([-1, X_train.shape[1]])
# plt.show()


X_validation1['is_remittance_final']=Y_validation
# X_validation1.to_csv("C:\Users\shubham.kamal\Desktop\LITM\wrong_predictions.csv",encoding='cp1256')


# X_train=data[data['check_checkNumber']!=40702][['row_noOfCharacters','remittance_result','total_digits','total_letters','total_spaces','total_others','distance_of_remittance_row']].reset_index(drop=True)
# Y_train=data[data['check_checkNumber']!=40702]['is_remittance_final'].reset_index(drop=True)
# X_validation=data[data['check_checkNumber']==40702][['row_noOfCharacters','remittance_result','total_digits','total_letters','total_spaces','total_others','distance_of_remittance_row']].reset_index(drop=True)
# Y_validation=data[data['check_checkNumber']==40702]['is_remittance_final'].reset_index(drop=True)
# tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english',vocabulary=vocab)
# theString = tfidf.fit_transform(data[data['check_checkNumber']!=40702]['row_string'])
# theString2 = tfidf.transform(data[data['check_checkNumber']==40702]['row_string'])
# combine1 = pd.DataFrame(theString.todense())
# combine1.columns = tfidf.get_feature_names()
# print(combine1.columns)
# combine2 = pd.DataFrame(theString2.todense())
# combine2.columns = tfidf.get_feature_names()
# print(combine2.columns)
# X_train = pd.concat([X_train, combine1.reset_index(drop=True)], axis=1, ignore_index=True)
# X_validation = pd.concat([X_validation, combine2.reset_index(drop=True)], axis=1, ignore_index=True)
# rfc = RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

df3=pd.DataFrame()
X_validation1['is_remittance_final']=Y_validation
X_validation1['predictions']=predictions
for i in range(0,X_validation.shape[0]):
    if Y_validation[i]!=predictions[i]:
        df3=df3.append(X_validation1.iloc[[i]],ignore_index=True)

#df3.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\wrong_cases.csv")
print('Started...')
data['remittance_result_2']=0
for i in data['check_checkNumber'].unique():
    print('i',i)
    for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
        print('j', j)
        temp=pd.DataFrame()
        temp1=pd.DataFrame()
        temp2=pd.DataFrame()
        temp=data[(data['check_checkNumber']==i) & (data['page_pageNumber']==j)]
        temp=temp.reset_index(drop=True,inplace=True)
        first_row=1
        last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
        temp = temp.sort_values('row_rowNumber', inplace=True)
        heading_row_number=first_row
        total_row_number=last_row
        for k in range(0,temp.shape[0]-1):
            print('k', k)
            if ((temp.at[k,'is_heading']==1) & (temp.at[k+1,'is_heading']==0)):
                temp1=temp1.append(temp.iloc[[k]],ignore_index=True)
            else:
                continue
        for k in range(1, temp.shape[0] - 1):
            print('k', k)
            if ((temp.at[k-1, 'is_total_final'] == 0) & (temp.at[k, 'is_total_final'] == 1)):
                temp2 = temp2.append(temp.iloc[[k]], ignore_index=True)
            else:
                continue
        if temp1.empty | temp2.empty:
            data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (
            (data['row_rowNumber'] > heading_row_number) & (
            data['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
        else:
            temp1 = temp1.sort_values('row_rowNumber')
            temp1 = temp1.reset_index(drop=True)
            temp2 = temp2.sort_values('row_rowNumber')
            temp2 = temp2.reset_index(drop=True)
            count1 = temp1.shape[0]
            count2 = temp2.shape[0]
            count3 = 0
            count4 = 0
            for k in range(0, count1):
                if ((temp1.at[count3, 'row_rowNumber'] < temp2.at[count4, 'row_rowNumber']) & (temp2.at[count4, 'row_rowNumber'] < temp1.at[count3 + 1, 'row_rowNumber'])):
                    data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] > temp1.at[count3, 'row_rowNumber']) & (data['row_rowNumber'] < temp2.at[count4, 'row_rowNumber'])), 'remittance_result_2'] = 1
                    count3 = count3 + 1
                    count4 = count4 + 1
                elif ((temp1.at[count3, 'row_rowNumber'] < temp2.at[count4, 'row_rowNumber']) & (
                    temp2.at[count4, 'row_rowNumber'] > temp1.at[count3 + 1, 'row_rowNumber'])):
                    count3 = count3 + 1
                elif (temp1.at[count3, 'row_rowNumber'] > temp2.at[count4, 'row_rowNumber']):
                    count4 = count4 + 1
                if ((count3 == count1) | (count4 == count2)):
                    break
                else:
                    continue
