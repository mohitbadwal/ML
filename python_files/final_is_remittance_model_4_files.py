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


####################################################################################################


data_set=pd.read_csv(r"D:\New folder\Modspace\CSVs\COMBINED_CSV.csv", sep=',',encoding='cp1256')
#data2=pd.read_csv(r"D:\New folder\Modspace\CSVs\COMBINED_CSV_test.csv", sep=',',encoding='cp1256',low_memory=False)

#data2=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Success_rows_final1.csv",sep=',',encoding='cp1256')
#data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Not_Success_rows_ver_clean.csv",sep=',',encoding='cp1256')


data_set=data_set[data_set['page_pageType']=='REMITTANCE_PAGE']
#data2=data2[data2['page_pageType']=='REMITTANCE_PAGE']
data=pd.DataFrame()
data2=pd.DataFrame()
diff_checks=data_set[data_set['is_remittance_final']!=data_set['is_remittance_final_original']]['check_checkNumber'].unique()

for i in data_set['check_checkNumber'].unique():
    if i not in diff_checks:
        data=data.append(data_set[data_set['check_checkNumber']==i],ignore_index=True)
    else:
        data2 = data2.append(data_set[data_set['check_checkNumber'] == i], ignore_index=True)

data=data.reset_index(drop=True)
data2=data2.reset_index(drop=True)

# data.to_csv("D:\\New folder\\Modspace\\CSVs\\COMBINED_CSV_training_new.csv")
#data2.to_csv("D:\\New folder\\Modspace\\CSVs\\Combined_test_Gaurav_Heading_Model2.csv")
data2=pd.read_csv(r"D:\New folder\Modspace\CSVs\Combined_test_Gaurav_Heading_Model2.csv", sep=',',encoding='cp1256')

suraj_success=pd.read_csv(r"D:\New folder\Modspace\CSVs\Success_Modspace_Suraj.csv", sep=',',encoding='cp1256')

#data=data.append(suraj_success,ignore_index=True)
data.to_csv("D:\\New folder\\Modspace\\CSVs\\COMBINED_CSV_training_new.csv")
# final model hai tera yea ?
# hahahahhs
# data.loc[data['page_type_final']=='check','page_type']=0
# data.loc[data['page_type_final']=='envelope','page_type']=1
# data.loc[(data['page_type_final']!='check') & (data['page_type_final']!='envelope'),'page_type']=2
# data=data.reset_index(drop=True)

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
            heading_row_number=df2.reset_index(drop=True).at[df2.shape[0]-1,'row_rowNumber']
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
            heading_row_number=df2.reset_index(drop=True).at[df2.shape[0]-1,'row_rowNumber']
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
cols=['count_of_features','ref_no_bool','date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_distanceFromLeft','row_distanceFromTop']

X_train=data[cols]
X_validation=df3[cols]
Y_train = data['is_remittance_final'].reset_index(drop=True)
Y_validation = df3['is_remittance_final'].reset_index(drop=True)


rfc = RandomForestClassifier(n_estimators=170,max_depth=15,random_state=42)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
predictions_prob=rfc.predict_proba(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

df3['predictions']=predictions

for i in range(0,df3.shape[0]):
    df3.at[i,'pred_proba_0']=predictions_prob[i][0]
    df3.at[i, 'pred_proba_1'] = predictions_prob[i][1]


df3=df3[['ratio_row_section','total_digits_coded','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_fetched','ref_no_bool','amount_col_man','date_flag','remittance_result','is_heading','is_total_final','count_of_features','is_remittance_final','predictions','is_remittance_final_original','pred_proba_0','pred_proba_1']]
df3.to_csv("D:\\New folder\\Modspace\\CSVs\\not_success_remittance_pages2.csv")
#df3.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\success_21.csv")
print(predictions_prob)

# next_test=pd.DataFrame()
# diff_checks2=df3[df3['is_remittance_final']!=df3['predictions']]['check_checkNumber'].unique()
#
# for i in df3['check_checkNumber'].unique():
#     if i in diff_checks2:
#         next_test=next_test.append(df3[df3['check_checkNumber']==i],ignore_index=True)
#     else:
#         data=data.append(df3[df3['check_checkNumber']==i],ignore_index=True)
#
#
#
#
# X_train=data[cols]
# X_validation=next_test[cols]
# Y_train = data['is_remittance_final'].reset_index(drop=True)
# Y_validation = next_test['is_remittance_final'].reset_index(drop=True)
#
#
# rfc = RandomForestClassifier(n_estimators=170,max_depth=15,random_state=42)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# predictions_prob=rfc.predict_proba(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
# next_test['predictions']=predictions
#
# for i in range(0,next_test.shape[0]):
#     next_test.at[i,'pred_proba_0']=predictions_prob[i][0]
#     next_test.at[i, 'pred_proba_1'] = predictions_prob[i][1]
#
#
# next_test=next_test[['ratio_row_section','total_digits_coded','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_fetched','ref_no_bool','amount_col_man','date_flag','remittance_result','is_heading','is_total_final','count_of_features','is_remittance_final','predictions','is_remittance_final_original','pred_proba_0','pred_proba_1']]
# next_test.to_csv("D:\\New folder\\Modspace\\CSVs\\not_success_remittance_pages.csv")





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
# print('Total OCRs : ',next_test['check_checkNumber'].unique().shape[0])
# print('Total OCRs mispredicted : ',next_test[next_test['is_remittance_final']!=next_test['predictions']]['check_checkNumber'].unique().shape[0])


print('Total OCRs mispredicted in terms of counts: ',to_check['check_checkNumber'].unique().shape[0])
print('Total OCRs mispredicted in non-remittance pages in terms of counts: ',to_check[to_check['page_type']!=2]['check_checkNumber'].unique().shape[0])
print('Total OCRs mispredicted in remittance pages in terms of counts: ',to_check[to_check['page_type']==2]['check_checkNumber'].unique().shape[0])
to_check.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\to_check1.csv")

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
temp2.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\to_check21.csv")
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

#
#
#
# ###############################################################################################
#
#
# data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Not_Success_rows_ver_clean.csv", sep=',',encoding='cp1256')
# data2=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\toKamal-1.4_not.csv", sep=',',encoding='cp1256',low_memory=False)
#
# data = data[data['page_type_final'] == 'remittance']
#
#
# data3=data.append(data2,ignore_index=True)
# data3['row_noOfCharacters']=pd.cut(data3['row_noOfCharacters'],bins=10).cat.codes
# data3=data3.reset_index(drop=True)
#
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
# data['row_noOfCharacters']=data3['row_noOfCharacters'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['row_noOfCharacters']=data3['row_noOfCharacters'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
#
# for i in data['check_checkNumber'].unique():
#     for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
#         temp=pd.DataFrame()
#         temp=data[(data['check_checkNumber']==i) & (data['page_pageNumber']==j)]
#         temp.reset_index(drop=True,inplace=True)
#         first_row=1
#         last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
#         df=pd.DataFrame()
#         df=temp[temp['is_total_final']==1]
#         if df.empty:
#             total_row_number=last_row
#         else:
#             total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
#         df2=pd.DataFrame()
#         df2=temp[temp['is_heading']==1]
#         if df2.empty:
#             heading_row_number=first_row
#         else:
#             heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber']<= heading_row_number),'remittance_result']=0
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] > heading_row_number) & (data['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0
#
# for i in data2['check_checkNumber'].unique():
#     for j in data2[data2['check_checkNumber']==i]['page_pageNumber'].unique():
#         temp=pd.DataFrame()
#         temp=data2[(data2['check_checkNumber']==i) & (data2['page_pageNumber']==j)]
#         temp.reset_index(drop=True,inplace=True)
#         first_row=1
#         last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
#         df=pd.DataFrame()
#         df=temp[temp['is_total_final']==1]
#         if df.empty:
#             total_row_number=last_row
#         else:
#             total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
#         df2=pd.DataFrame()
#         df2=temp[temp['is_heading']==1]
#         if df2.empty:
#             heading_row_number=first_row
#         else:
#             heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber']<= heading_row_number),'remittance_result']=0
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & ((data2['row_rowNumber'] > heading_row_number) & (data2['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0
#
#
# for i in range(0,data3.shape[0]):
#     s=data3.at[i,'row_string']
#     digits = sum(c.isdigit() for c in s)
#     letters = sum(c.isalpha() for c in s)
#     spaces = sum(c.isspace() for c in s)
#     others = len(s) - digits - letters - spaces
#
#     total_charac=digits+letters+spaces+others
#     data3.at[i,'total_digits'] = digits/total_charac*100
#     data3.at[i, 'total_letters'] = letters/total_charac*100
#     data3.at[i, 'total_spaces'] = spaces/total_charac*100
#     data3.at[i, 'total_others'] = others/total_charac*100
#
# data3['total_digits_coded']=pd.cut(data3['total_digits'],bins=10).cat.codes
# data3['total_letters_coded']=pd.cut(data3['total_letters'],bins=10).cat.codes
# data3['total_spaces_coded']=pd.cut(data3['total_spaces'],bins=10).cat.codes
# data3['total_others_coded']=pd.cut(data3['total_others'],bins=10).cat.codes
# data3=data3.reset_index(drop=True)
#
#
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
# data['total_digits_coded']=data3['total_digits_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_digits_coded']=data3['total_digits_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_letters_coded']=data3['total_letters_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_letters_coded']=data3['total_letters_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_spaces_coded']=data3['total_spaces_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_spaces_coded']=data3['total_spaces_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_others_coded']=data3['total_others_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_others_coded']=data3['total_others_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
#
# df3=pd.DataFrame()
# df3=data2.reset_index(drop=True)
#
#
# data['ratio_row_section']=data['row_noOfCharacters']/data['section_noOfCharacters']
# df3['ratio_row_section']=df3['row_noOfCharacters']/df3['section_noOfCharacters']
#
#
# data['amount_col_man']=0
# for i in range(0,data.shape[0]):
#     s=data.at[i,'row_string']
#     if '$' in s:
#         data.at[i, 'amount_col_man'] = 1
#     s = s.replace(',', '')
#     s=s.replace('$',' ')
#     digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
#     for j in digits:
#         if float(j)<=data.at[i,'check_checkAmount']:
#             data.at[i,'amount_col_man']=1
#             break
#
#
# df3['amount_col_man']=0
# for i in range(0,df3.shape[0]):
#     s=df3.at[i,'row_string']
#     if '$' in s:
#         df3.at[i, 'amount_col_man'] = 1
#     s = s.replace(',', '')
#     s = s.replace('$', ' ')
#     digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
#     for j in digits:
#         if float(j)<=df3.at[i,'check_checkAmount']:
#             df3.at[i,'amount_col_man']=1
#             break
#
#
# pattern=re.compile("Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))",re.IGNORECASE)
#
#
# def dateFlag(x):
#     global pattern
#     if pattern.search(str(x)) is not None:
#         return 1
#     else:
#         return 0
#
#
# data['date_flag'] = data['row_string'].apply(dateFlag)
# df3['date_flag'] = df3['row_string'].apply(dateFlag)
#
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
#
#
#
# def is_date(string):
#     try:
#         parse(string)
#         return 1
#     except ValueError:
#         return 0
#
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
#
#
#
#
# # X_train=data[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# # X_validation=df3[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# # Y_train = data['is_remittance_final'].reset_index(drop=True)
# # Y_validation = df3['is_remittance_final'].reset_index(drop=True)
#
# X=data[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# Y= data['is_remittance_final'].reset_index(drop=True)
# validation_size=0.3
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
#
#
# rfc = RandomForestClassifier(n_estimators=300)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
#
#
#
# # df3['predictions']=predictions
# # df3=df3[['ratio_row_section','total_digits_coded','total_others_coded','is_heading','is_total_final','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_col_man','date_flag','remittance_result','is_remittance_final','predictions','ocr_filepath']]
# # df3.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\not_success_2.csv")
# # #
# #
# # ############################################################################
#
# data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Not_Success_rows_ver_clean.csv", sep=',',encoding='cp1256')
# data2=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\toKamal-1.3_success.csv", sep=',',encoding='cp1256',low_memory=False)
#
# data = data[(data['page_type_final'] == 'remittance') | (data['page_type_final'] == 'others')]
#
# data3=data.append(data2,ignore_index=True)
# data3['row_noOfCharacters']=pd.cut(data3['row_noOfCharacters'],bins=10).cat.codes
# data3=data3.reset_index(drop=True)
#
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
# data['row_noOfCharacters']=data3['row_noOfCharacters'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['row_noOfCharacters']=data3['row_noOfCharacters'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
#
# for i in data['check_checkNumber'].unique():
#     for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
#         temp=pd.DataFrame()
#         temp=data[(data['check_checkNumber']==i) & (data['page_pageNumber']==j)]
#         temp.reset_index(drop=True,inplace=True)
#         first_row=1
#         last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
#         df=pd.DataFrame()
#         df=temp[temp['is_total_final']==1]
#         if df.empty:
#             total_row_number=last_row
#         else:
#             total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
#         df2=pd.DataFrame()
#         df2=temp[temp['is_heading']==1]
#         if df2.empty:
#             heading_row_number=first_row
#         else:
#             heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber']<= heading_row_number),'remittance_result']=0
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] > heading_row_number) & (data['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0
#
# for i in data2['check_checkNumber'].unique():
#     for j in data2[data2['check_checkNumber']==i]['page_pageNumber'].unique():
#         temp=pd.DataFrame()
#         temp=data2[(data2['check_checkNumber']==i) & (data2['page_pageNumber']==j)]
#         temp.reset_index(drop=True,inplace=True)
#         first_row=1
#         last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
#         df=pd.DataFrame()
#         df=temp[temp['is_total_final']==1]
#         if df.empty:
#             total_row_number=last_row
#         else:
#             total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
#         df2=pd.DataFrame()
#         df2=temp[temp['is_heading']==1]
#         if df2.empty:
#             heading_row_number=first_row
#         else:
#             heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber']<= heading_row_number),'remittance_result']=0
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & ((data2['row_rowNumber'] > heading_row_number) & (data2['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0
#
#
# for i in range(0,data3.shape[0]):
#     s=data3.at[i,'row_string']
#     digits = sum(c.isdigit() for c in s)
#     letters = sum(c.isalpha() for c in s)
#     spaces = sum(c.isspace() for c in s)
#     others = len(s) - digits - letters - spaces
#
#     total_charac=digits+letters+spaces+others
#     data3.at[i,'total_digits'] = digits/total_charac*100
#     data3.at[i, 'total_letters'] = letters/total_charac*100
#     data3.at[i, 'total_spaces'] = spaces/total_charac*100
#     data3.at[i, 'total_others'] = others/total_charac*100
#
# data3['total_digits_coded']=pd.cut(data3['total_digits'],bins=10).cat.codes
# data3['total_letters_coded']=pd.cut(data3['total_letters'],bins=10).cat.codes
# data3['total_spaces_coded']=pd.cut(data3['total_spaces'],bins=10).cat.codes
# data3['total_others_coded']=pd.cut(data3['total_others'],bins=10).cat.codes
# data3=data3.reset_index(drop=True)
#
#
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
# data['total_digits_coded']=data3['total_digits_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_digits_coded']=data3['total_digits_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_letters_coded']=data3['total_letters_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_letters_coded']=data3['total_letters_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_spaces_coded']=data3['total_spaces_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_spaces_coded']=data3['total_spaces_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_others_coded']=data3['total_others_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_others_coded']=data3['total_others_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
#
# df3=pd.DataFrame()
# df3=data2.reset_index(drop=True)
#
#
# data['ratio_row_section']=data['row_noOfCharacters']/data['section_noOfCharacters']
# df3['ratio_row_section']=df3['row_noOfCharacters']/df3['section_noOfCharacters']
#
#
# data['amount_col_man']=0
# for i in range(0,data.shape[0]):
#     s=data.at[i,'row_string']
#     if '$' in s:
#         data.at[i, 'amount_col_man'] = 1
#     s = s.replace(',', '')
#     s=s.replace('$',' ')
#     digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
#     for j in digits:
#         if float(j)<=data.at[i,'check_checkAmount']:
#             data.at[i,'amount_col_man']=1
#             break
#
#
# df3['amount_col_man']=0
# for i in range(0,df3.shape[0]):
#     s=df3.at[i,'row_string']
#     if '$' in s:
#         df3.at[i, 'amount_col_man'] = 1
#     s = s.replace(',', '')
#     s = s.replace('$', ' ')
#     digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
#     for j in digits:
#         if float(j)<=df3.at[i,'check_checkAmount']:
#             df3.at[i,'amount_col_man']=1
#             break
#
#
# pattern=re.compile("Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))",re.IGNORECASE)
#
#
# def dateFlag(x):
#     global pattern
#     if pattern.search(str(x)) is not None:
#         return 1
#     else:
#         return 0
#
#
# data['date_flag'] = data['row_string'].apply(dateFlag)
# df3['date_flag'] = df3['row_string'].apply(dateFlag)
#
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
#
# def is_date(string):
#     try:
#         parse(string)
#         return 1
#     except ValueError:
#         return 0
#
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
#
#
# X_train=data[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# X_validation=df3[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# Y_train = data['is_remittance_final'].reset_index(drop=True)
# Y_validation = df3['is_remittance_final'].reset_index(drop=True)
# voca=['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_rowNumber']
# rfc = RandomForestClassifier(n_estimators=300)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
# importances = rfc.feature_importances_
# print(importances)
# std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# print("Feature ranking:")
# #print(train_features.columns)
#
# for f in range(X_train.shape[1]):
#     print("%d. %s (%f)" % (f + 1, voca[indices[f]], importances[indices[f]]))
#
# # # Plot the feature importances of the forest
# # plt.figure()
# # plt.title("Feature importances")
# # plt.bar(range(X_train.shape[1]), importances[indices],
# #        color="r", yerr=std[indices], align="center")
# # plt.xticks(range(X_train.shape[1]), myList)
# # plt.xlim([-1, X_train.shape[1]])
# # plt.show()
#
#
#
# df3['predictions']=predictions
# df3=df3[['ratio_row_section','total_digits_coded','total_others_coded','is_heading','is_total_final','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_col_man','date_flag','remittance_result','is_remittance_final','predictions','ocr_filepath']]
# df3.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\success_1.csv")
#
#
# ##############################################################################################
#
# data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Not_Success_rows_ver_clean.csv", sep=',',encoding='cp1256')
# data2=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\toKamal-1.4_success.csv", sep=',',encoding='cp1256',low_memory=False)
#
#
# data=data[data['page_type_final']=='remittance']
#
# data3=data.append(data2,ignore_index=True)
# data3['row_noOfCharacters']=pd.cut(data3['row_noOfCharacters'],bins=10).cat.codes
# data3=data3.reset_index(drop=True)
#
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
# data['row_noOfCharacters']=data3['row_noOfCharacters'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['row_noOfCharacters']=data3['row_noOfCharacters'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
#
# for i in data['check_checkNumber'].unique():
#     for j in data[data['check_checkNumber']==i]['page_pageNumber'].unique():
#         temp=pd.DataFrame()
#         temp=data[(data['check_checkNumber']==i) & (data['page_pageNumber']==j)]
#         temp.reset_index(drop=True,inplace=True)
#         first_row=1
#         last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
#         df=pd.DataFrame()
#         df=temp[temp['is_total_final']==1]
#         if df.empty:
#             total_row_number=last_row
#         else:
#             total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
#         df2=pd.DataFrame()
#         df2=temp[temp['is_heading']==1]
#         if df2.empty:
#             heading_row_number=first_row
#         else:
#             heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber']<= heading_row_number),'remittance_result']=0
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & ((data['row_rowNumber'] > heading_row_number) & (data['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
#         data.loc[(data['check_checkNumber'] == i) & (data['page_pageNumber'] == j) & (data['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0
#
# for i in data2['check_checkNumber'].unique():
#     for j in data2[data2['check_checkNumber']==i]['page_pageNumber'].unique():
#         temp=pd.DataFrame()
#         temp=data2[(data2['check_checkNumber']==i) & (data2['page_pageNumber']==j)]
#         temp.reset_index(drop=True,inplace=True)
#         first_row=1
#         last_row=temp.at[temp.shape[0]-1,'row_rowNumber']
#         df=pd.DataFrame()
#         df=temp[temp['is_total_final']==1]
#         if df.empty:
#             total_row_number=last_row
#         else:
#             total_row_number=df.reset_index(drop=True).at[0,'row_rowNumber']
#         df2=pd.DataFrame()
#         df2=temp[temp['is_heading']==1]
#         if df2.empty:
#             heading_row_number=first_row
#         else:
#             heading_row_number=df2.reset_index(drop=True).at[0,'row_rowNumber']
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber']<= heading_row_number),'remittance_result']=0
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & ((data2['row_rowNumber'] > heading_row_number) & (data2['row_rowNumber'] < total_row_number)), 'remittance_result'] = 1
#         data2.loc[(data2['check_checkNumber'] == i) & (data2['page_pageNumber'] == j) & (data2['row_rowNumber'] >= total_row_number), 'remittance_result'] = 0
#
#
# for i in range(0,data3.shape[0]):
#     s=data3.at[i,'row_string']
#     digits = sum(c.isdigit() for c in s)
#     letters = sum(c.isalpha() for c in s)
#     spaces = sum(c.isspace() for c in s)
#     others = len(s) - digits - letters - spaces
#
#     total_charac=digits+letters+spaces+others
#     data3.at[i,'total_digits'] = digits/total_charac*100
#     data3.at[i, 'total_letters'] = letters/total_charac*100
#     data3.at[i, 'total_spaces'] = spaces/total_charac*100
#     data3.at[i, 'total_others'] = others/total_charac*100
#
# data3['total_digits_coded']=pd.cut(data3['total_digits'],bins=10).cat.codes
# data3['total_letters_coded']=pd.cut(data3['total_letters'],bins=10).cat.codes
# data3['total_spaces_coded']=pd.cut(data3['total_spaces'],bins=10).cat.codes
# data3['total_others_coded']=pd.cut(data3['total_others'],bins=10).cat.codes
# data3=data3.reset_index(drop=True)
#
#
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
# data['total_digits_coded']=data3['total_digits_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_digits_coded']=data3['total_digits_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_letters_coded']=data3['total_letters_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_letters_coded']=data3['total_letters_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_spaces_coded']=data3['total_spaces_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_spaces_coded']=data3['total_spaces_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
# data['total_others_coded']=data3['total_others_coded'].loc[:data.shape[0]-1].reset_index(drop=True)
# data2['total_others_coded']=data3['total_others_coded'].loc[data.shape[0]:data3.shape[0]-1].reset_index(drop=True)
# data=data.reset_index(drop=True)
# data2=data2.reset_index(drop=True)
#
#
# df3=pd.DataFrame()
# df3=data2.reset_index(drop=True)
#
#
# data['ratio_row_section']=data['row_noOfCharacters']/data['section_noOfCharacters']
# df3['ratio_row_section']=df3['row_noOfCharacters']/df3['section_noOfCharacters']
#
#
# data['amount_col_man']=0
# for i in range(0,data.shape[0]):
#     s=data.at[i,'row_string']
#     if '$' in s:
#         data.at[i, 'amount_col_man'] = 1
#     s = s.replace(',', '')
#     s=s.replace('$',' ')
#     digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
#     for j in digits:
#         if float(j)<=data.at[i,'check_checkAmount']:
#             data.at[i,'amount_col_man']=1
#             break
#
#
# df3['amount_col_man']=0
# for i in range(0,df3.shape[0]):
#     s=df3.at[i,'row_string']
#     if '$' in s:
#         df3.at[i, 'amount_col_man'] = 1
#     s = s.replace(',', '')
#     s = s.replace('$', ' ')
#     digits=re.findall(r"\s+\d+\.\d+$|\s+\d+\.\d+\s+", s,flags=re.MULTILINE)
#     for j in digits:
#         if float(j)<=df3.at[i,'check_checkAmount']:
#             df3.at[i,'amount_col_man']=1
#             break
#
#
# pattern=re.compile("Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?\s+\d{1,2}[,/.]\s+\d{4}([0-3]?[0-9][.|/][0-1]?[0-9][.|/](([0-9]{4})|([0-9]{2})))|([0-1]?[0-9][.|/][0-3]?[0-9][.|/](([0-9]{4})|([0-9]{2})))",re.IGNORECASE)
#
#
# def dateFlag(x):
#     global pattern
#     if pattern.search(str(x)) is not None:
#         return 1
#     else:
#         return 0
#
#
# data['date_flag'] = data['row_string'].apply(dateFlag)
# df3['date_flag'] = df3['row_string'].apply(dateFlag)
#
#
#
#
# def is_date(string):
#     try:
#         parse(string)
#         return 1
#     except ValueError:
#         return 0
#
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
#
#
#
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
#
# X_train=data[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','total_others_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# X_validation=df3[['date_flag','amount_col_man','ratio_row_section','row_noOfCharacters','remittance_result','total_digits_coded','total_others_coded','row_JosasisLRCoordinates_left','row_JosasisLRCoordinates_right','row_distanceFromLeft','row_distanceFromTop']]
# Y_train = data['is_remittance_final'].reset_index(drop=True)
# Y_validation = df3['is_remittance_final'].reset_index(drop=True)
#
# rfc = RandomForestClassifier(n_estimators=300)
# rfc.fit(X_train, Y_train)
# predictions = rfc.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
#
#
# df3['predictions']=predictions
# df3=df3[['ratio_row_section','total_digits_coded','total_others_coded','is_heading','is_total_final','check_checkAmount','check_checkNumber','page_pageNumber','row_rowNumber','row_string','amount_col_man','date_flag','remittance_result','is_remittance_final','predictions','ocr_filepath']]
# df3.to_csv("C:\\Users\\shubham.kamal\\Desktop\\LITM\\success_2.csv")
#
#
# ######################################################################
#
# #
# data=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Not_Success_rows_ver_clean.csv", sep=',',encoding='cp1256')
# # data2=pd.read_csv(r"C:\Users\shubham.kamal\Desktop\LITM\Status.csv", sep=',',encoding='cp1256',low_memory=False)
# # print(data.shape[0])
# # print(data2.shape[0])
# # for i in data['check_checkNumber'].unique():
# #     data.loc[data['check_checkNumber']==i,'indexing_status'] = data2[data2['Check Number']==i]['indexing_status'].values
# # print(data['indexing_status'].value_counts())
# #
# temp=pd.DataFrame()
# temp=data.groupby(['check_checkNumber','page_pageNumber']).size().reset_index().rename(columns={0:'count'})
# print(temp.head(4))
# print(temp.shape[0])

print(1&1)


data=pd.read_csv(r"D:\New folder\Modspace\CSVs\Modspace_MOHIT.csv", sep=',',encoding='cp1256')
data1=pd.read_csv(r"D:\New folder\Modspace\CSVs\Modspace_SHUBHAM.csv", sep=',',encoding='cp1256')
data2=pd.read_csv(r"D:\New folder\Modspace\CSVs\Modspace_SURAJ.csv", sep=',',encoding='cp1256')
data3=pd.read_csv(r"D:\New folder\Modspace\CSVs\UNILEVER_GAURAV.csv", sep=',',encoding='cp1256')

data=data.append(data1,ignore_index=True)
data=data.append(data2,ignore_index=True)
data=data.append(data3,ignore_index=True)

data.to_csv("D:\\New folder\\Modspace\\CSVs\\COMBINED_CSV.csv")