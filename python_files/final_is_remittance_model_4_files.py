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
#data.loc[data['page_type_final']=='check','page_type']=0
#data.loc[data['page_type_final']=='envelope','page_type']=1
#data.loc[(data['page_type_final']!='check') & (data['page_type_final']!='envelope'),'page_type']=2
data=data.reset_index(drop=True)

data3=data.append(data2,ignore_index=True)
data3['row_noOfCharacters']=pd.cut(data3['row_noOfCharacters'],bins=10).cat.codes
data3['row_distanceFromLeft']=pd.cut(data3['row_distanceFromLeft'],bins=10)
print('row_distanceFromLeft',data3['row_distanceFromLeft'].unique())
data3['row_distanceFromLeft']=data3['row_distanceFromLeft'].cat.codes
data3['row_distanceFromTop']=pd.cut(data3['row_distanceFromTop'],bins=10)
print('row_distanceFromTop',data3['row_distanceFromTop'].unique())
data3['row_distanceFromTop']=data3['row_distanceFromTop'].cat.codes
data3['row_JosasisLRCoordinates_left']=pd.cut(data3['row_JosasisLRCoordinates_left'],bins=10).cat.codes
data3['row_JosasisLRCoordinates_right']=pd.cut(data3['row_JosasisLRCoordinates_right'],bins=10).cat.codes
data3=data3.reset_index(drop=True)

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

data['row_JosasisLRCoordinates_left']=data3['row_JosasisLRCoordinates_left'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['row_JosasisLRCoordinates_left']=data3['row_JosasisLRCoordinates_left'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

data['row_JosasisLRCoordinates_right']=data3['row_JosasisLRCoordinates_right'].loc[:data.shape[0]-1].reset_index(drop=True)
data2['row_JosasisLRCoordinates_right']=data3['row_JosasisLRCoordinates_right'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
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
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] >= heading_row_number) & (data['row_rowNumber'] <= total_row_number)), 'remittance_result'] = 1
        data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] > total_row_number), 'remittance_result'] = 0

data.to_csv("just_aise_hi.csv")



data2['remittance_result']=0
for i in data2['check_checkNumber'].unique():
    for j in data2[data2['check_checkNumber']==i]['page_pageNumber'].unique():
        temp=pd.DataFrame()
        temp=data2[(data2['check_checkNumber']==i) & (data2['page_pageNumber']==j)]
        temp.reset_index(drop=True,inplace=True)
        first_row=1
        last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
        df = pd.DataFrame()
        df = temp[temp['is_total'] == 1]
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
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & ((data2['row_rowNumber'] >= heading_row_number) & (data2['row_rowNumber'] <= total_row_number)), 'remittance_result'] = 1
        data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber'] > total_row_number), 'remittance_result'] = 0


for i in range(0,data3.shape[0]):
    s=data3.at[i,'row_string']
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    spaces = sum(c.isspace() for c in s)
    others = len(s) - digits - letters - spaces

    total_charac=digits+letters+others
    data3.at[i,'total_digits'] = digits/total_charac*100
    data3.at[i, 'total_letters'] = letters/total_charac*100
    data3.at[i, 'total_spaces'] = spaces/total_charac*100
    data3.at[i, 'total_others'] = others/total_charac*100

data3['total_digits_coded']=pd.cut(data3['total_digits'],bins=10)
print('bins = ',data3['total_digits_coded'])
data3['total_digits_coded']=data3['total_digits_coded'].cat.codes
data3['total_letters_coded']=pd.cut(data3['total_letters'],bins=10).cat.codes
data3['total_spaces_coded']=pd.cut(data3['total_spaces'],bins=10).cat.codes
data3['total_others_coded']=pd.cut(data3['total_others'],bins=10).cat.codes
data3=data3.reset_index(drop=True)


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


df3=pd.DataFrame()
df3=data2.reset_index(drop=True)


data['ratio_row_section']=data['row_noOfCharacters']/data['section_noOfCharacters']
df3['ratio_row_section']=df3['row_noOfCharacters']/df3['section_noOfCharacters']


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


df3['amount_col_man']=0
for i in range(0,df3.shape[0]):
    s=df3.at[i,'row_string']
    if '$' in s:
        df3.at[i, 'amount_col_man'] = 1
    s = s.replace('$', ' ')
    s=s.replace(', ',',')
    digits=re.findall(r"\d+\.\d+|\d{1,2}[\,]{1}\d{1,3}[\.]{1}\d{1,2}", s,flags=re.MULTILINE)
    max=0
    flag=0
    for j in digits:
        j=j.replace(',','')
        if ((float(j)<=df3.at[i,'check_checkAmount']) & (float(j)>0)):
            flag=1
            if (max<float(j)):
                max=float(j)
    if flag==1:
        df3.at[i,'amount_col_man']=1
        df3.at[i, 'amount_fetched'] = max


data['ref_no_bool']=0
for i in range(0,data.shape[0]):
    s=data.at[i,'row_string']
    #s = s.replace('-', '')
    digits=re.findall(r"[0-9]+", s,flags=re.MULTILINE)
    for j in digits:
        if len(j)>=6:
            data.at[i,'ref_no_bool']=1
            break


df3['ref_no_bool']=0
for i in range(0,df3.shape[0]):
    s=df3.at[i,'row_string']
    #s = s.replace('-', '')
    digits=re.findall(r"[0-9]+", s,flags=re.MULTILINE)
    for j in digits:
        if len(j)>=6:
            df3.at[i,'ref_no_bool']=1
            break

pattern=re.compile("Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|\d{1,4}[\-|\.|\/]{1}\d{1,4}[\-|\.|\/]{1}\d{2,4}",re.IGNORECASE)


def dateFlag(x):
    global pattern
    if pattern.search(str(x)) is not None:
        return 1
    else:
        return 0


data['date_flag'] = data['row_string'].apply(dateFlag)
df3['date_flag'] = df3['row_string'].apply(dateFlag)


data['count_of_features']=data['date_flag']+data['amount_col_man']+data['ref_no_bool']+data['remittance_result']
df3['count_of_features']=df3['date_flag']+df3['amount_col_man']+df3['ref_no_bool']+df3['remittance_result']
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

# for i in range(0,df3.shape[0]):
#     s=df3.at[i,'row_string']
#     words=s.split()
#     for j in words:
#         try:
#             if is_date(j)==1:
#                 df3.at[i, 'date_flag'] = 1
#                 break
#         except OverflowError:
#             continue
#
# for i in range(0,data.shape[0]):
#     s=data.at[i,'row_string']
#     words=s.split()
#     for j in words:
#         try:
#             if is_date(j)==1:
#                 data.at[i, 'date_flag'] = 1
#                 break
#         except OverflowError:
#             continue



# X=data[data['page_type_final']=='remittance'][['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# Y= data[data['page_type_final']=='remittance']['is_remittance_final'].reset_index(drop=True)

# X=pd.DataFrame()
# Y=pd.DataFrame()
# count=0
# for i in data['check_checkNumber'].unique():
#     X=data[data['']]

def function_threshold(predictions, predictions_prob, Y_validation, thresh_list):
    output_classes_ = Y_validation.unique()
    combined_predictions = pd.DataFrame(
        {'pred_label': predictions, 'actual_label': Y_validation, 'prob_0': predictions_prob[:, 0],
         'prob_1': predictions_prob[:, 1]})
    combined_predictions['max_prob'] = combined_predictions[['prob_0', 'prob_1']].max(axis=1)
    final_threshold_predictions = pd.DataFrame()

    for class_ in output_classes_:
        thresh = thresh_list[class_]
        pred_class_ = combined_predictions[combined_predictions['pred_label'] == class_]
        pred_class_thresh_ = pred_class_[pred_class_['max_prob'] > thresh]
        final_threshold_predictions = final_threshold_predictions.append(pred_class_thresh_)

    return final_threshold_predictions, final_threshold_predictions.loc[:,
                                        'actual_label'], final_threshold_predictions.loc[:,
                                                         'pred_label'], final_threshold_predictions.loc[:,
                                                                        ['prob_0', 'prob_1']]


def print_metrics(Y_validation, predictions, predictions_prob=None):
    print("Accuracy ", accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    if predictions_prob is not None:
        print("Log loss", sklearn.metrics.log_loss(Y_validation, predictions_prob))


cols=['page_type','date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']
cols=['ref_no_bool','date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_distanceFromLeft','row_distanceFromTop']

X_train=data[cols]
X_validation=df3[cols]
Y_train = data['is_remittance_final'].reset_index(drop=True)
Y_validation = df3['is_remittance_final'].reset_index(drop=True)


#X_validation.to_csv('C:\\Users\\shubham.kamal\\Desktop\\LITM\\test1_saad.csv')

rfc = RandomForestClassifier(n_estimators=170,max_depth=15,random_state=42)
rfc.fit(X_train, Y_train)
#X_validation['remittance_result']=1
predictions = rfc.predict(X_validation)
predictions_prob=rfc.predict_proba(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#
# thresh=[0.1,0.2,0.3,0.4]
# for i in thresh:
#     combined_predictions ,threshold_actual , threshold_predictions , threshold_prob=function_threshold(predictions,predictions_prob,Y_validation,thresh_list=[0.5,i])
#     print("Lost samples:", 1 - (len(threshold_predictions)/len(predictions)))
#     print_metrics(threshold_actual,threshold_predictions,threshold_prob)
#     print(0.5,i)
#     print("\n","\n")
#     print('******************************************',"\n")
#
# thresh=[0.9,0.8,0.7,0.6]
# for i in thresh:
#     combined_predictions ,threshold_actual , threshold_predictions , threshold_prob=function_threshold(predictions,predictions_prob,Y_validation,thresh_list=[i,0.5])
#     print("Lost samples:", 1 - (len(threshold_predictions)/len(predictions)))
#     print_metrics(threshold_actual,threshold_predictions,threshold_prob)
#     print(i,0.5)
#     print("\n","\n")
#     print('******************************************',"\n")
# combined_predictions ,threshold_actual , threshold_predictions , threshold_prob=function_threshold(predictions,predictions_prob,Y_validation,thresh_list=[0.7,0.1])
# print("Lost samples:", 1 - (len(threshold_predictions)/len(predictions)))
# print_metrics(threshold_actual,threshold_predictions,threshold_prob)

# temp=pd.DataFrame()
# temp=data[data['page_type_final']=='others'][['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# temp2=pd.DataFrame()
# temp2=data[data['page_type_final']=='others']['is_remittance_final'].reset_index(drop=True)
# X_validation=X_validation.append(temp,ignore_index=True)
# Y_validation=Y_validation.append(temp2,ignore_index=True)
# X_validation=X_validation.reset_index(drop=True)
# Y_validation=Y_validation.reset_index(drop=True)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

df3['predictions']=predictions
# df3['pred_proba_0']=predictions_prob[0]
# df3['pred_proba_1']=predictions_prob[1]

# df3['amount_sum']=1
# for i in df3['check_checkNumber'].unique():
#     print(i)
#     Total_amount=0
#     temp=df3[(df3['check_checkNumber']==i) & (df3['amount_col_man']==1)]
#     temp['amount_sum']=0
#     temp.reset_index(drop=True, inplace=True)
#     if temp.empty:
#         continue
#     check_amount=temp.at[0,'check_checkAmount']
#     print(check_amount)
#     for k in range(0,temp.shape[0]):
#         Total_amount=Total_amount+temp.at[k,'amount_fetched']
#         temp.at[k,'amount_sum']=1
#         if Total_amount==check_amount:
#             break
#     df3.loc[(df3['check_checkNumber'] == i) & (df3['amount_col_man'] == 1),'amount_sum']=temp['amount_sum']
#
# df3['final_label']=df3['predictions'] & df3['amount_sum']
#
#
# predictions=df3['final_label']
# Y_validation = df3['is_remittance_final'].reset_index(drop=True)
# print(accuracy_score(Y_validation, predictions ))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


#df3=df3[['ratio_row_section','total_digits_coded','page_type','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_fetched','ref_no_bool','amount_col_man','date_flag','remittance_result','is_heading','is_total_final','count_of_features','is_remittance_final','predictions','ocr_filepath']]
df3.to_csv(test_file)
print(predictions_prob)

print(df3[df3['is_remittance_final']!=df3['predictions']]['check_checkNumber'].unique())




importances = rfc.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
#print(train_features.columns)

for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))


print('Total OCRs : ',df3['check_checkNumber'].unique().shape[0])
df3=df3.reset_index(drop=True)
to_check=pd.DataFrame()
flag1=0
flag2=0
flag3=0
count_original=0
count_predicted=0
for i in df3['check_checkNumber'].unique():
    for j in df3[df3['check_checkNumber']==i]['page_pageNumber'].unique():
        temp=pd.DataFrame()
        temp=df3[(df3['check_checkNumber']==i) & (df3['page_pageNumber']==j)]
        temp.reset_index(drop=True,inplace=True)
        count_original = temp[temp['is_remittance_final']==1].shape[0]
        count_predicted = temp[temp['predictions']==1].shape[0]
        if count_original!=count_predicted:
            to_check=to_check.append(temp,ignore_index=True)
            to_check=to_check.reset_index(drop=True)

print('Total OCRs mispredicted : ',df3[df3['is_remittance_final']!=df3['predictions']]['check_checkNumber'].unique().shape[0])
print('Total OCRs mispredicted - remittances: ',df3[(df3['is_remittance_final']!=df3['predictions']) & (df3['page_type']==2)]['check_checkNumber'].unique().shape[0])
print('Total OCRs mispredicted - other than remittances: ',df3[(df3['is_remittance_final']!=df3['predictions']) & (df3['page_type']!=2)]['check_checkNumber'].unique().shape[0])

print('Total OCRs mispredicted in terms of counts: ',to_check['check_checkNumber'].unique().shape[0])
print('Total OCRs mispredicted in non-remittance pages in terms of counts: ',to_check[to_check['page_type']!=2]['check_checkNumber'].unique().shape[0])
print('Total OCRs mispredicted in remittance pages in terms of counts: ',to_check[to_check['page_type']==2]['check_checkNumber'].unique().shape[0])
to_check.to_csv("to_check1.csv")

temp2=pd.DataFrame()
flag1=0
flag2=0
flag3=0
for i in to_check['check_checkNumber'].unique():
    temp = pd.DataFrame()
    temp = to_check[to_check['check_checkNumber'] == i]
    temp.reset_index(drop=True, inplace=True)
    count_original = temp[temp['is_remittance_final'] == 1].shape[0]
    count_predicted = temp[temp['predictions'] == 1].shape[0]
    if ((count_original == 0) & (count_predicted > 0)):
        flag1 = flag1 + 1
    elif ((count_original > 0) & (count_predicted == 0)):
        flag2 = flag2 + 1
    elif ((count_original > 0) & (count_predicted > 0) & (count_original != count_predicted)):
        flag3 = flag3 + 1
        temp2=temp2.append(temp,ignore_index=True)
temp2=temp2.reset_index(drop=True)
temp2.to_csv("to_check21.csv")
print('\n','At an OCR level - All page types are concerned')
print('actual = 0  predicted > 0  :', flag1)
print('actual > 0  predicted = 0  :', flag2)
print('actual > 0  predicted > 0  but actual != predicted  :', flag3)
print('Total = ',flag1+flag2+flag3)

flag1=0
flag2=0
flag3=0
for i in to_check['check_checkNumber'].unique():
    temp = pd.DataFrame()
    temp = to_check[(to_check['check_checkNumber'] == i) & (to_check['page_type']==2)]
    temp.reset_index(drop=True, inplace=True)
    count_original = temp[temp['is_remittance_final'] == 1].shape[0]
    count_predicted = temp[temp['predictions'] == 1].shape[0]
    if ((count_original == 0) & (count_predicted > 0)):
        flag1 = flag1 + 1
    elif ((count_original > 0) & (count_predicted == 0)):
        flag2 = flag2 + 1
    elif ((count_original > 0) & (count_predicted > 0) & (count_original != count_predicted)):
        flag3 = flag3 + 1
print('\n','At an OCR level - only remittance pages are considered')
print('actual = 0  predicted > 0  :', flag1)
print('actual > 0  predicted = 0  :', flag2)
print('actual > 0  predicted > 0  but actual != predicted  :', flag3)
print('Total = ',flag1+flag2+flag3)

