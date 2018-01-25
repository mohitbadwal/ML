import pandas as pd
import random
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
from dateutil.parser import parse

data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Not_Success_rows_ver_clean.csv", sep=',',encoding='cp1256')
data2=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\test_new.csv", sep=',',encoding='cp1256',low_memory=False)


data = data[data['page_type_final'] == 'remittance']
data2=data2[data2['page_pageType']=='REMITTANCE_PAGE']

data3=data.append(data2,ignore_index=True)
data3['row_noOfCharacters']=pd.cut(data3['row_noOfCharacters'],bins=10).cat.codes
data3=data3.reset_index(drop=True)
print('data3',data3[['row_noOfCharacters','page_pageNumber']].head(5))

data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)
data['row_noOfCharacters']=data3['row_noOfCharacters'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['row_noOfCharacters']=data3['row_noOfCharacters'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)


print('data',data[['row_noOfCharacters','page_pageNumber']].head(5))
print('data2',data2[['row_noOfCharacters','page_pageNumber']].head(5))

def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]

def cleanandstem(sentence):
    return cleaning(sentence)

for i in data['check_checkNumber'].unique():
    for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
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

for i in data2['check_checkNumber'].unique():
    #total_pages=temp.at[0,'check_accountNumber']
    for j in data2[data2['check_checkNumber']==i]['page_pageNumber'].unique():
        temp=pd.DataFrame()
        temp=data2[(data2['check_checkNumber']==i) & (data2['page_pageNumber']==j)]
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
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber']<= heading_row_number),'remittance_result']=0
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & ((data2['row_rowNumber'] > heading_row_number) & (data2['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0


data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)


for i in range(0,data.shape[0]):
    s=data.at[i,'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces
    #alpha=letters
    #print(s,digits,letters,spaces,others)
    total_charac=digits+letters+spaces+others
    data.at[i,'total_digits'] = digits/total_charac*100
    data.at[i, 'total_letters'] = letters/total_charac*100
    data.at[i, 'total_spaces'] = spaces/total_charac*100
    data.at[i, 'total_others'] = others/total_charac*100
    #print(digits,':',alpha,'  ',data.at[i,'row_numberAlphaRatio'])

for i in range(0,data2.shape[0]):
    s=data2.at[i,'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces
    #alpha=letters
    #print(s,digits,letters,spaces,others)
    total_charac=digits+letters+spaces+others
    data2.at[i,'total_digits'] = digits/total_charac*100
    data2.at[i, 'total_letters'] = letters/total_charac*100
    data2.at[i, 'total_spaces'] = spaces/total_charac*100
    data2.at[i, 'total_others'] = others/total_charac*100
#print(data['distance_of_remittance_row'])


for i in range(0,data3.shape[0]):
    s=data3.at[i,'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces
    #alpha=letters
    #print(s,digits,letters,spaces,others)
    total_charac=digits+letters+spaces+others
    data3.at[i,'total_digits'] = digits/total_charac*100
    data3.at[i, 'total_letters'] = letters/total_charac*100
    data3.at[i, 'total_spaces'] = spaces/total_charac*100
    data3.at[i, 'total_others'] = others/total_charac*100

data3['total_digits_coded']=pd.cut(data3['total_digits'],bins=10).cat.codes
data3['total_letters_coded']=pd.cut(data3['total_letters'],bins=10).cat.codes
data3['total_spaces_coded']=pd.cut(data3['total_spaces'],bins=10).cat.codes
data3['total_others_coded']=pd.cut(data3['total_others'],bins=10).cat.codes
data3=data3.reset_index(drop=True)


# for i in range(0,data3.shape[0]):
#     count=data3.at[i,'total_digits']
#     if (count<31.4):
#         data3.at[i,'total_digits_coded']=0
#     else:
#         data3.at[i, 'total_digits_coded'] =1
#
# for i in range(0,data3.shape[0]):
#     count=data3.at[i,'total_others']
#     if (count<4.54545454545):
#         data3.at[i,'total_others_coded']=0
#     else:
#         data3.at[i, 'total_others_coded'] =1



data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)
data['total_digits_coded']=data3['total_digits_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['total_digits_coded']=data3['total_digits_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

data['total_letters_coded']=data3['total_letters_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['total_letters_coded']=data3['total_letters_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

data['total_spaces_coded']=data3['total_spaces_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['total_spaces_coded']=data3['total_spaces_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

data['total_others_coded']=data3['total_others_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['total_others_coded']=data3['total_others_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)


print('data2 unique',data2['check_checkNumber'].unique().shape)
df3=pd.DataFrame()
print('shape',data2.shape[0])
data2=data2.reset_index(drop=True)
for i in data2['check_checkNumber'].unique():
    count=0
    for j in data2[data2['check_checkNumber']==i]['page_pageNumber'].unique():
        temp=pd.DataFrame()
        temp=data2[(data2['check_checkNumber']==i) & (data2['page_pageNumber']==j)]
        temp=temp.reset_index(drop=True)
        for k in range(0,temp.shape[0]):
            if temp.at[k,'is_remittance_final']==1:
                count=count+1
                for l in range(0,temp.shape[0]):
                    df3=df3.append(temp.iloc[[l]],ignore_index=True)
                break
    if count==0:
        print(i)
df3=df3.reset_index(drop=True)
print('df3 unique',df3['check_checkNumber'].unique().shape)
vocab=['inst','policy'
#'lpm',#'pckg'#'construction','bond'
]
df3['renB_newB']=0
for i in range(0,df3.shape[0]):
    s=df3.at[i,'row_string']
    if(('renB' in s) | ('newB' in s) | ('endt' in s) | ('cfee' in s) | ('cnl' in s) | ('flat' in s) | ('NEWB' in s) | ('RENB' in s) | ('ENDT' in s) | ('CFEE' in s) | ('FLAT' in s)):
        df3.at[i,'renB_newB']=1

data['renB_newB'] = 0
for i in range(0, data.shape[0]):
    s = data.at[i, 'row_string']
    if (('renb' in s) | ('newB' in s) | ('endt' in s) | ('cfee' in s) | ('cnl' in s) | ('flat' in s) | ('NEWB' in s) | ('RENB' in s) | ('ENDT' in s) | ('CFEE' in s) | ('FLAT' in s)):
         data.at[i, 'renB_newB'] = 1



# def is_date(string):
#     try:
#         parse(string)
#         return 1
#     except ValueError:
#         return 0
#
# df3['date_boolean']=0
# for i in range(0,df3.shape[0]):
#     s=df3.at[i,'row_string']
#     words=s.split()
#     for j in words:
#         try:
#             if is_date(j)==1:
#                 df3.at[i, 'date_boolean'] = 1
#                 break
#         except OverflowError:
#             print('String too long!')
#             continue
#
# data['date_boolean'] = 0
# for i in range(0, data.shape[0]):
#     s = data.at[i, 'row_string']
#     words = s.split()
#     for j in words:
#         try:
#             if is_date(j) == 1:
#                 data.at[i, 'date_boolean'] = 1
#                 break
#         except OverflowError:
#             print('String too long!')
#             continue

data['ratio_row_section']=data['row_noOfCharacters']/data['section_noOfCharacters']
df3['ratio_row_section']=df3['row_noOfCharacters']/df3['section_noOfCharacters']
# tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english',vocabulary=vocab)
# theString = tfidf.fit_transform(data['row_string'])
# combine1 = pd.DataFrame(theString.todense())
# combine1.columns = tfidf.get_feature_names()
# print(combine1.columns)

data['amount_col_man']=0
for i in range(0,data.shape[0]):
    s=data.at[i,'row_string']
    if '$' in s:
        data.at[i, 'amount_col_man'] = 1
    s = s.replace(',', '')
    s=s.replace('$',' ')
    digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
    for j in digits:
        if float(j)<=data.at[i,'check_checkAmount']:
            data.at[i,'amount_col_man']=1
            break


df3['amount_col_man']=0
for i in range(0,df3.shape[0]):
    s=df3.at[i,'row_string']
    if '$' in s:
        df3.at[i, 'amount_col_man'] = 1
    s = s.replace(',', '')
    s = s.replace('$', ' ')
    digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
    for j in digits:
        if float(j)<=df3.at[i,'check_checkAmount']:
            df3.at[i,'amount_col_man']=1
            break


pattern=re.compile("Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))",re.IGNORECASE)


def dateFlag(x):
    global pattern
    if pattern.search(str(x)) is not None:
        return 1
    else:
        return 0


data['date_flag'] = data['row_string'].apply(dateFlag)
df3['date_flag'] = df3['row_string'].apply(dateFlag)

data['date_amt_combined']=0
df3['date_amt_combined']=0
data.loc[(data['date_flag']==1) & (data['amount_col_man']==1),'date_amt_combined']=1
data.loc[(data['date_flag']==0) & (data['amount_col_man']==1),'date_amt_combined']=1
data.loc[(data['date_flag']==1) & (data['amount_col_man']==0),'date_amt_combined']=0
data.loc[(data['date_flag']==0) & (data['amount_col_man']==0),'date_amt_combined']=0

df3.loc[(df3['date_flag']==1) & (df3['amount_col_man']==1),'date_amt_combined']=1
df3.loc[(df3['date_flag']==0) & (df3['amount_col_man']==1),'date_amt_combined']=1
df3.loc[(df3['date_flag']==1) & (df3['amount_col_man']==0),'date_amt_combined']=0
df3.loc[(df3['date_flag']==0) & (df3['amount_col_man']==0),'date_amt_combined']=0




# tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english',vocabulary=vocab)
# theString = tfidf.fit_transform(data['row_string'])
# theString2 = tfidf.transform(df3['row_string'])
# combine1 = pd.DataFrame(theString.todense())
# combine1.columns = tfidf.get_feature_names()
# print(combine1.columns)
# combine2 = pd.DataFrame(theString2.todense())
# combine2.columns = tfidf.get_feature_names()
# print(combine2.columns)

X_train=data[['date_amt_combined','date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','total_others_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
X_validation=df3[['date_amt_combined','date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','total_others_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
#,'section_sectionNumber','section_joasisLRCoordinates_left','section_joasisLRCoordinates_right','section_joasisTBCoordinates_bottom','section_joasisTBCoordinates_top','section_noOfCharacters','section_noOfRows'
voca=['date_amt_combined','date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','total_others_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']
#X_train=pd.concat([X_train, combine1.reset_index(drop=True)], axis=1, ignore_index=True)
#X_validation = pd.concat([X_validation, combine2.reset_index(drop=True)], axis=1, ignore_index=True)
Y_train = data['is_remittance_final'].reset_index(drop=True)
Y_validation = df3['is_remittance_final'].reset_index(drop=True)

# from sklearn.neural_network import MLPClassifier
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))






df3['predictions']=predictions
#data[['ratio_row_section','row_string','renB_newB','date_boolean']].to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\just_to_check.csv")
df3=df3[['ratio_row_section','renB_newB','total_digits_coded','total_letters_coded','total_spaces_coded','total_others_coded','total_digits','total_letters','total_spaces','total_others','is_heading','is_total_final','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_col_man','date_flag','date_amt_combined','remittance_result','is_remittance_final','predictions','ocr_filepath']]
df3.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\Not_success_1.csv")

#
# validation_size = 0.3
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_train, Y_train, test_size=validation_size)
#
# rfc = RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
#
#
# train=pd.DataFrame()
# test=pd.DataFrame()
# count=0
# for i in data['check_checkNumber'].unique():
#     if count<100:
#         test=test.append(data[data['check_checkNumber']==i],ignore_index=True)
#         count=count+1
#     else:
#         train=train.append(data[data['check_checkNumber']==i],ignore_index=True)
#         count=count+1
#
# train=train.reset_index(drop=True)
# test=test.reset_index(drop=True)
# X_train=train[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','total_others_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# Y_train=train['is_remittance_final']
# X_validation=test[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','total_others_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# Y_validation=test['is_remittance_final']
#
# rfc = RandomForestClassifier(n_estimators=200)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
# test['predictions']=predictions
# #data[['ratio_row_section','row_string','renB_newB','date_boolean']].to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\just_to_check.csv")
# test=test[['ratio_row_section','renB_newB','total_digits_coded','total_letters_coded','total_spaces_coded','total_others_coded','total_digits','total_letters','total_spaces','total_others','is_heading','is_total_final','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_col_man','date_flag','remittance_result','is_remittance_final','predictions','ocr_filepath']]
# test.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\test_100_OCR_Level.csv")
















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
    print("%d. %s (%f)" % (f + 1, voca[indices[f]], importances[indices[f]]))
    myList.append(voca[indices[f]])

# # Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), myList)
plt.xlim([-1, X_train.shape[1]])
plt.show()








# print('DIGITS')
# count=0
# for i in range(0,10):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_digits_coded']==i)):
#             count=count+1
#         if (df3.at[j,'is_remittance_final']==1):
#             count2=count2+1
#     print(i,'=',(count/count2)*100,'%')
# print('\n')
#
#
# print('LETTERS')
# for i in range(0,10):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_letters_coded']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print(i,'=',count/count2*100,'%')
# print('\n')
#
# print('SPACES')
# for i in range(0,10):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_spaces_coded']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print(i,'=',count/count2*100,'%')
# print('\n')
#
# print('OTHERS')
# for i in range(0,10):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_others_coded']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print(i,'=',count/count2*100,'%')
# print('\n')
#
#
#
# print('DIGITS')
# count=0
# for i in range(0,10):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_digits_coded']==i)):
#             count=count+1
#         if (df3.at[j,'is_remittance_final']==1):
#             count2=count2+1
#     print('Bin ', i, '  Remittance Lines= ', count, ' Total lines= ', df3[df3['total_digits_coded'] == i].shape[0],' P%= ',count/df3[df3['total_digits_coded'] == i].shape[0]*100,'%')
# print('\n')
#
# print('LETTERS')
# for i in range(0,10):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_letters_coded']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print('Bin ',i, '  Remittance Lines= ', count, 'Total lines= ',df3[df3['total_letters_coded']==i].shape[0],' P%= ',count/df3[df3['total_letters_coded'] == i].shape[0]*100,'%')
# print('\n')
#
#
# print('OTHERS')
# for i in range(0,9):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_others_coded']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print('Bin ',i, '  Remittance Lines= ', count, 'Total lines= ',df3[df3['total_others_coded']==i].shape[0],' P%= ',count/df3[df3['total_others_coded'] == i].shape[0]*100,'%')
# print('\n')
#
#
#
# print('SPACES')
# for i in range(0,7):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_spaces_coded']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print('Bin ', i, '  Remittance Lines= ', count, 'Total lines= ', df3[df3['total_spaces_coded'] == i].shape[0],' P%= ',count/df3[df3['total_spaces_coded'] == i].shape[0]*100,'%')
# print('\n')
# print('DIGITS')
# count=0
# count2=0
# df3.sort_values('total_digits',inplace=True)
# df3=df3.reset_index(drop=True)
# print('min=',df3.at[0,'total_digits'])
# for i in range(0,df3.shape[0]):
#     df3.at[i,'total_digits_new']=count
#     count2=count2+1
#     if count2==1293:
#         count2=0
#         count=count+1
#         print('max=',df3.at[i,'total_digits'],' BIN ',df3.at[i,'total_digits_new'])
#         if i!=df3.shape[0]-1:
#             print('\n','min=',df3.at[i+1,'total_digits'])
# print('\n')
# for i in range(0,df3.shape[0]):
#     count=df3.at[i,'total_digits_new']
#     if ((count==0) | (count==1) | (count==2)):
#         df3.at[i,'total_digits_new']=0
#     else:
#         df3.at[i, 'total_digits_new'] =1
#
#
#
#
# print('LETTERS')
# count=0
# count2=0
# df3.sort_values('total_letters',inplace=True)
# df3=df3.reset_index(drop=True)
# print('min=',df3.at[0,'total_letters'])
# for i in range(0,df3.shape[0]):
#     df3.at[i,'total_letters_new']=count
#     count2=count2+1
#     if count2==1293:
#         count2=0
#         count=count+1
#         print('max=',df3.at[i,'total_letters'],' BIN ',df3.at[i,'total_letters_new'])
#         if i!=df3.shape[0]-1:
#             print('\n','min=',df3.at[i+1,'total_letters'])
# print('\n')
#
# print('SPACES')
# count=0
# count2=0
# df3.sort_values('total_spaces',inplace=True)
# df3=df3.reset_index(drop=True)
# print('min=',df3.at[0,'total_spaces'])
# for i in range(0,df3.shape[0]):
#     df3.at[i,'total_spaces_new']=count
#     count2=count2+1
#     if count2==1293:
#         count2=0
#         count=count+1
#         print('max=',df3.at[i,'total_spaces'],' BIN ',df3.at[i,'total_spaces_new'])
#         if i!=df3.shape[0]-1:
#             print('\n','min=',df3.at[i+1,'total_spaces'])
# print('\n')
#
# print('OTHERS')
# count=0
# count2=0
# df3.sort_values('total_others',inplace=True)
# df3=df3.reset_index(drop=True)
# print('min=',df3.at[0,'total_others'])
# for i in range(0,df3.shape[0]):
#     df3.at[i,'total_others_new']=count
#     count2=count2+1
#     if count2==1293:
#         count2=0
#         count=count+1
#         print('max=',df3.at[i,'total_others'],' BIN ',df3.at[i,'total_others_new'])
#         if i!=df3.shape[0]-1:
#             print('\n','min=',df3.at[i+1,'total_others'])
# print('\n')
#
#
# print('DIGITS')
# count=0
# for i in range(0,5):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_digits_new']==i)):
#             count=count+1
#         if (df3.at[j,'is_remittance_final']==1):
#             count2=count2+1
#     print('Bin ', i, '  Remittance Lines= ', count, '  Total lines= ', df3[df3['total_digits_new'] == i].shape[0],' P%= ',count/df3[df3['total_digits_new'] == i].shape[0]*100,'%')
# print('\n')
#
# print('LETTERS')
# for i in range(0,5):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_letters_new']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print('Bin ',i, '  Remittance Lines= ', count, '  Total lines= ',df3[df3['total_letters_new']==i].shape[0],' P%= ',count/df3[df3['total_letters_new'] == i].shape[0]*100,'%')
# print('\n')
#
#
# print('OTHERS')
# for i in range(0,5):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_others_new']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print('Bin ',i, '  Remittance Lines= ', count, '  Total lines= ',df3[df3['total_others_new']==i].shape[0],' P%= ',count/df3[df3['total_others_new'] == i].shape[0]*100,'%')
# print('\n')
#
#
#
# print('SPACES')
# for i in range(0,5):
#     count=0
#     count2=0
#     for j in range(0,df3.shape[0]):
#         if ((df3.at[j,'is_remittance_final']==1) & (df3.at[j,'total_spaces_new']==i)):
#             count=count+1
#         if (df3.at[j, 'is_remittance_final'] == 1):
#             count2=count2+1
#     print('Bin ', i, '  Remittance Lines= ', count, '  Total lines= ', df3[df3['total_spaces_new'] == i].shape[0],' P%= ',count/df3[df3['total_spaces_new'] == i].shape[0]*100,'%')
# print('\n')
