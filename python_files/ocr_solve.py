import string

import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


# this function is used to remove punctuations marks and also removes stop words like 'the' , 'a' ,'an'
def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]


# applying both together
def cleanandstem(sentence):
    return cleaning(sentence)


dataset = pd.read_csv('Mohit_new.csv')
s = ""
is_remit_flag = 0
is_total_flag = 0
# replace yes to 1 and no to 0
dataset['is_remittance'].replace("yes", 1, inplace=True)
dataset['is_total'].replace("yes", 1, inplace=True)
dataset['is_remittance'].replace("no", 0, inplace=True)
dataset['is_total'].replace("no", 0, inplace=True)


def func(x):
    global s
    global is_remit_flag
    global is_total_flag
    s = s + " " + x['row_string']
    is_remit_flag = is_remit_flag or x['is_remittance']
    is_total_flag = is_total_flag or x['is_total']


temp_dataset = pd.DataFrame()

dr1 = dataset.groupby("check_checkNumber")

for name, group in dr1:
    dr2 = group.groupby("page_type")
    for name2, group2 in dr2:
        s = ""
        is_remit_flag = 0
        is_total_flag = 0
        group2.apply(func, axis=1)
        temp = pd.DataFrame({"check_amount": group2['check_checkAmount'].values[0],
                             "check_number": group2['check_checkNumber'].values[0],
                             "check_pages": group2['check_noOfPages'].values[0],
                             "check_page_number": group2['page_pageNumber'].values[0],
                             "is_remittance": is_remit_flag,
                             "is_total": is_total_flag,
                             "string": s, "page_type": name2}, index=[0])
        temp_dataset = pd.concat([temp_dataset, temp], ignore_index=True, axis=0)

# TODO: make line parser
# temp_dataset.to_csv("ocr_new_file.csv", index=False)
# Using countVectorizer to get word counts
import numpy as np
temp_dataset.replace(np.NaN,0,inplace=True)
# temp_dataset = temp_dataset[(temp_dataset['page_type'] == 'check') | (temp_dataset['page_type'] == 'envelope')]

# splitting training and testing data
temp_dataset_train = temp_dataset.loc[:len(temp_dataset)*0.8]
temp_dataset_test = temp_dataset.loc[len(temp_dataset)*0.8:]
countVectorizer = CountVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.5, stop_words='english')
theString = countVectorizer.fit_transform(temp_dataset_train['string'])
combine1 = pd.DataFrame(theString.todense())
# combine1.rename(columns=countVectorizer.get_feature_names(), inplace=True)
combine1.columns = countVectorizer.get_feature_names()
print(combine1.columns)

X = temp_dataset_train.loc[:, ['check_page_number', 'check_pages']]
X = pd.concat([combine1.reset_index(drop=True), X.reset_index(drop=True)], axis=1,ignore_index=True)
Y = temp_dataset_train.loc[:, 'page_type']
validation_size = 0.2
seed = 10
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
# applying ML algos
df = temp_dataset_test[(temp_dataset_test['page_type'] == 'check') | (temp_dataset_test['page_type'] == 'envelope')]
print(df.shape)
theString = countVectorizer.transform(df['string'])
combine1 = pd.DataFrame(theString.todense())
# combine1.rename(columns=countVectorizer.get_feature_names(), inplace=True)
combine1.columns = countVectorizer.get_feature_names()
X_validation = df.loc[:, ['check_page_number', 'check_pages']]
X_validation = pd.concat([combine1.reset_index(drop=True), X_validation.reset_index(drop=True)], axis=1)
Y_validation = df.loc[:, 'page_type']
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X, Y)
print(rfc.feature_importances_)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
