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
data2=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Success_rows_final1.csv", sep=',',encoding='cp1256',low_memory=False)

data_set=data[data['page_pageType']=='REMITTANCE_PAGE']
data2=data2[data2['page_pageType']=='REMITTANCE_PAGE']

data3=data.append(data2,ignore_index=True)

data3['row_noOfCharacters']=pd.cut(data3['row_noOfCharacters'],bins=10).cat.codes
data3['row_distanceFromLeft']=pd.cut(data3['row_distanceFromLeft'],bins=10).cat.codes
data3['row_distanceFromTop']=pd.cut(data3['row_distanceFromTop'],bins=10).cat.codes


data3['ratio_row_section']=data3['row_noOfCharacters']/data3['section_noOfCharacters']
data3['ratio_row_section']=pd.cut(data3['ratio_row_section'],bins=10).cat.codes

data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)
data['row_noOfCharacters']=data3['row_noOfCharacters'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['row_noOfCharacters']=data3['row_noOfCharacters'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

data['row_distanceFromLeft']=data3['row_distanceFromLeft'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['row_distanceFromLeft']=data3['row_distanceFromLeft'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

data['row_distanceFromTop']=data3['row_distanceFromTop'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['row_distanceFromTop']=data3['row_distanceFromTop'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

data['ratio_row_section']=data3['ratio_row_section'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['ratio_row_section']=data3['ratio_row_section'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)


data['remittance_result']=0
for i in data['check_checkNumber'].unique():
    for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
        temp=pd.DataFrame()
        temp=data[(data['check_checkNumber']==i) & (data['page_pageNumber']==j)]
        temp.reset_index(drop=True,inplace=True)
        first_row=1
        last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
        df=pd.DataFrame()
        df=temp[temp['is_total_final']==1]
        df.sort_values('row_rowNumber', inplace=True)
        df=df.reset_index(drop=True)
        df2 = pd.DataFrame()
        df2 = temp[temp['is_heading'] == 1]
        df2.sort_values('row_rowNumber', inplace=True)
        df2 = df2.reset_index(drop=True)
        if ((df.empty) & (df2.empty)):
            continue
        if df.empty:
            total_row_number=last_row
        else:
            total_row_number=df.reset_index(drop=True).at[df.shape[0]-1,'row_rowNumber']
            data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber']== total_row_number), 'remittance_result'] = 1
        if df2.empty:
            heading_row_number=first_row
        else:
            heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber']< heading_row_number),'remittance_result']=0
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] > heading_row_number) & (data['row_rowNumber'] <= total_row_number)), 'remittance_result'] = 1
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] > total_row_number), 'remittance_result'] = 0



data2['remittance_result']=0
for i in data2['check_checkNumber'].unique():
    for j in data2[data2['check_checkNumber']==i]['page_pageNumber'].unique():
        temp=pd.DataFrame()
        temp=data2[(data2['check_checkNumber']==i) & (data2['page_pageNumber']==j)]
        temp.reset_index(drop=True,inplace=True)
        first_row=1
        last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
        df = pd.DataFrame()
        df = temp[temp['is_total_final'] == 1]
        df.sort_values('row_rowNumber', inplace=True)
        df = df.reset_index(drop=True)
        df2 = pd.DataFrame()
        df2 = temp[temp['is_heading'] == 1]
        df2.sort_values('row_rowNumber', inplace=True)
        df2 = df2.reset_index(drop=True)
        if ((df.empty) & (df2.empty)):
            continue
        if df.empty:
            total_row_number=last_row
        else:
            total_row_number=df.reset_index(drop=True).at[df.shape[0]-1,'row_rowNumber']
            data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (
            data2['row_rowNumber'] == total_row_number), 'remittance_result'] = 1
        if df2.empty:
            heading_row_number=first_row
        else:
            heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber']< heading_row_number),'remittance_result']=0
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & ((data2['row_rowNumber'] > heading_row_number) & (data2['row_rowNumber'] <= total_row_number)), 'remittance_result'] = 1
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber'] > total_row_number), 'remittance_result'] = 0

for i in range(0,data3.shape[0]):
    s=data3.at[i,'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces
    total_charac=digits+letters+others
    data3.at[i,'total_digits'] = digits/total_charac*100

data3['total_digits']=pd.cut(data3['total_digits'],bins=10).cat.codes


data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)
data['total_digits']=data3['total_digits'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['total_digits']=data3['total_digits'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)


data['amount_col_man']=0
for i in range(0,data.shape[0]):
    s=data.at[i,'row_string']
    if '$' in s:
        data.at[i, 'amount_col_man'] = 1
    s=s.replace('$',' ')
    s = s.replace(', ', ',')
    digits=re.findall(r"\d+\.\d+|\d{1,2}[\,]{1}\d{1,3}[\.]{1}\d{1,2}", s,flags=re.MULTILINE)
    max=0
    flag=0
    for j in digits:
        j=j.replace(',','')
        if ((float(j)<=data.at[i,'check_checkAmount']) & (float(j)>0)):
            flag = 1
            if (max < float(j)):
                max = float(j)
    if (flag == 1):
        data.at[i, 'amount_col_man'] = 1
        data.at[i, 'amount_fetched'] = max


data2['amount_col_man']=0
for i in range(0,data2.shape[0]):
    s=data2.at[i,'row_string']
    if '$' in s:
        data2.at[i, 'amount_col_man'] = 1
    s = s.replace('$', ' ')
    s=s.replace(', ',',')
    digits=re.findall(r"\d+\.\d+|\d{1,2}[\,]{1}\d{1,3}[\.]{1}\d{1,2}", s,flags=re.MULTILINE)
    max=0
    flag=0
    for j in digits:
        j=j.replace(',','')
        if ((float(j)<=data2.at[i,'check_checkAmount']) & (float(j)>0)):
            flag=1
            if (max<float(j)):
                max=float(j)
    if flag==1:
        data2.at[i,'amount_col_man']=1
        data2.at[i, 'amount_fetched'] = max


data['ref_no_bool']=0
for i in range(0,data.shape[0]):
    s=data.at[i,'row_string']
    #s = s.replace('-', '')
    digits=re.findall(r"[0-9]+", s,flags=re.MULTILINE)
    for j in digits:
        if len(j)>=6:
            data.at[i,'ref_no_bool']=1
            break


data2['ref_no_bool']=0
for i in range(0,data2.shape[0]):
    s=data2.at[i,'row_string']
    #s = s.replace('-', '')
    digits=re.findall(r"[0-9]+", s,flags=re.MULTILINE)
    for j in digits:
        if len(j)>=6:
            data2.at[i,'ref_no_bool']=1
            break


pattern=re.compile("Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|\d{1,4}[\-|\.|\/]{1}\d{1,4}[\-|\.|\/]{1}\d{2,4}",re.IGNORECASE)
def dateFlag(x):
    global pattern
    if pattern.search(str(x)) is not None:
        return 1
    else:
        return 0


data['date_flag'] = data['row_string'].apply(dateFlag)
data2['date_flag'] = data2['row_string'].apply(dateFlag)

data['count_of_features']=data['date_flag']+data['amount_col_man']+data['ref_no_bool']+data['remittance_result']
data2['count_of_features']=data2['date_flag']+data2['amount_col_man']+data2['ref_no_bool']+data2['remittance_result']

cols=['ref_no_bool','date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits','row_distanceFromLeft','row_distanceFromTop']

data[cols].to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\The_TRAINING_SET_FEATURES_VALUE.csv")

X_train=data[cols]
X_validation=data2[cols]
Y_train = data['is_remittance_final'].reset_index(drop=True)
Y_validation = data2['is_remittance_final'].reset_index(drop=True)

rfc = RandomForestClassifier(n_estimators=170,max_depth=15,random_state=42)
rfc.fit(X_train, Y_train)

predictions = rfc.predict(X_validation)
predictions_prob=rfc.predict_proba(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

data2['predictions']=predictions
for i in range(0,data2.shape[0]):
    data2.at[i,'pred_proba_0']=predictions_prob[i][0]
    data2.at[i, 'pred_proba_1'] = predictions_prob[i][1]


df3=data2[['row_distanceFromLeft','row_distanceFromTop','ratio_row_section','total_digits','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_fetched','amount_col_man','ref_no_bool','date_flag','remittance_result','is_heading','is_total_final','count_of_features','is_remittance_final','predictions','pred_proba_0','pred_proba_1']]
df3.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\THE_TESTING_SET_FEATURES_VALUE.csv")



print('Total OCRs : ',data2['check_checkNumber'].unique().shape[0])
print('Total OCRs mispredicted : ',data2[data2['is_remittance_final']!=data2['predictions']]['check_checkNumber'].unique().shape[0])

print('Mispredicted check Numbers: ','\n',data2[data2['is_remittance_final']!=data2['predictions']]['check_checkNumber'].unique())



#FEATURE IMPORTANCE
importances = rfc.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
#print(train_features.columns)

for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))

