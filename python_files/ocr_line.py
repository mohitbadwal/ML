import string
import re
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# what the heeeelll

# this function is used to remove punctuations marks and also removes stop words like 'the' , 'a' ,'an'
from sklearn.neural_network import MLPClassifier


def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]


def cleaning_new(sentence):
    fd = '?-+:)(;,'
    punctuation_removed = [char for char in sentence if char not in fd]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return ' '.join(l)


# applying both together
def cleanandstem(sentence):
    return cleaning(sentence)


'''
$45,342.45 $56,23.78
'''
pattern_number = re.compile('([$]?[0-9]*[\,]?[0-9]*[\.][0-9]+)')


def convertNumber(s):
    d = ''
    for i in str(s):
        if i != ',' and i != '$':
            d = d + i
    if len(d) > 0:
        return int(float(d))
    else:
        return -1


# function to identify numbers
def isNumber(s, e):
    li = str(s).split(" ")
    i = 0
    gh = 0
    d = 0
    f = -1
    er = []
    er1 = []
    for x in li:
        if pattern_number.fullmatch(x) is not None:
            er.append(x)
            er1.append(d)
            i = 1
            if f == -1:
                f = d
        d = d + 1

    er1.sort(reverse=True)
    # print(len(er1),li)
    for j in range(0, len(er1)):
        li.pop(er1[j])
    print(er, er1, i)
    if i == 1:
        f = 0
        for x in er:
            print("printing", x, e)
            if convertNumber(x) >= 0:  # int(float(e)):
                gh = 5
                # li = li[:f]
                print("here", e)
                li.append(x)
                f = 1
                break
        if f == 0:
            li.append(er[-1])
        return ' '.join(li), gh
    else:
        return s, gh


# pattern to match totals
pattern = re.compile(
    "(\^?[$]?[0-9]*[\,]?[0-9]*[\.][0-9]+\$?)|(.*((total)(s)?|(amount)).*([$]?([0-9]*[\,]?[0-9]*[\.][0-9]+)))")


# function for transformation of string to identify the possibility of a total according to the regular expression
def totalFlag(x):
    global pattern
    d = x['row_string']
    eddd = cleaning_new(d)
    e = x['check_checkAmount']
    s = str(eddd).lower().strip()
    print(s)
    s, gh = isNumber(s, e)
    print(s)
    # if gh == 0:
    if pattern.fullmatch(s) is not None:
        if re.search('(gross)', s) is None and re.search('(render)', s) is None \
                and re.search('(comm(\s)?%)|(com(\s)?%)', s) is None:
            if gh != 0:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        # print(str(x))
        return 0
        # else:
        #  return 1


dataset = pd.read_csv(r'D:\backup\PycharmProjects\test\Image '
                      r'Batches-20171017T131547Z-001\Not_Success_rows_ver_clean.csv',
                      encoding='cp1256')

dataset_test = pd.read_csv(r'D:\backup\PycharmProjects\test\Image '
                           r'Batches-20171017T131547Z-001\test_data_merged_row_level.csv',
                           encoding='cp1256')

s = ""
is_remit_flag = 0
is_total_flag = 0


# replace yes to 1 and no to 0
# dataset['is_remittance'].replace("yes", 1, inplace=True)
# dataset['is_total'].replace("yes", 1, inplace=True)
# dataset['is_remittance'].replace("no", 0, inplace=True)
# dataset['is_total'].replace("no", 0, inplace=True)

# replaces last 3 rows with 1
def last(x):
    for i in range(0, len(x)):
        if x.loc[i]['row_isLastRow'] == 1:
            if x.loc[i]['page_noOfRows'] > 3:
                x.loc[i - 1, 'row_isLastRow'] = 1
                x.loc[i - 2, 'row_isLastRow'] = 1
                print(x.loc[i - 1]['row_isLastRow'])
    return x


dataset = dataset[dataset['page_type_final'] == 'remittance'].reset_index()
dataset_test = dataset_test[(dataset_test['pred'] == 2) | (dataset_test['pred'] == 3)].reset_index()
print(dataset.shape)

# countVectorizer = CountVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.5, stop_words='english')
# theString = countVectorizer.fit_transform(dataset['row_string'])

dataset['rows'] = dataset['page_noOfRows'] - dataset['row_rowNumber']
dataset['total'] = dataset.apply(totalFlag, axis=1)
dataset = last(dataset)

dataset_test['rows'] = dataset_test['page_noOfRows'] - dataset_test['row_rowNumber']
dataset_test['total'] = dataset_test.apply(totalFlag, axis=1)
dataset_test = last(dataset_test)

tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=100, stop_words='english', vocabulary=
{'total', 'totals', 'grand', 'check'})
theString = tfidf.fit_transform(dataset['row_string'])
# from sklearn.externals import joblib
# joblib.dump(tfidf,"tfidf_ocr_total.pkl")
# tfidf = joblib.load("tfidf_ocr_total.pkl")
theTestString = tfidf.transform(dataset_test['row_string'])

combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()

combine2 = pd.DataFrame(theTestString.todense())
combine2.columns = tfidf.get_feature_names()
print(combine1.columns)
X = dataset.loc[:, [  # 'row_distanceFromTop',
    'rows',
    'total',
    'row_isLastRow',
    'row_string']]
X = pd.concat([combine1.reset_index(drop=True), X.reset_index(drop=True)], axis=1, ignore_index=True)
Y = dataset.loc[:, 'is_total_final']

X_test = dataset_test.loc[:, [  # 'row_distanceFromTop',
    'rows',
    'total',
    'row_isLastRow',
    'row_string']]
X_test = pd.concat([combine2.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1, ignore_index=True)

X = X.iloc[:, :-1]
er = X_test.iloc[:, -1]
X_test = X_test.iloc[:, :-1]


# Y_validation = Y_validation.iloc[-10:-9]
# TODO: Get func(x) in export branch for combining the regex model with ML model
def func(x):
    if x['total'] == 1 and x['pred_proba_0'] < 0.88 and x['is_total_final'] == 0:
        return 1
    return x['is_total_final']


rfc = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu')
rfc.fit(X, Y)
# print(rfc.feature_importances_)
predictions = rfc.predict(X_test)
predictions_prob = rfc.predict_proba(X_test)
pred_prob = pd.DataFrame(data=predictions_prob, columns=[0, 1])
det = pd.DataFrame({"str": er.values, "total":
    X_test.copy(deep=False).iloc[:, -2].values, "is_total_final": predictions, "pred_proba_0": pred_prob[0],
                    "pred_proba_1": pred_prob[1]})
det.to_csv("det1.csv")
det['pred'] = det.apply(func, axis=1)

a4 = pd.DataFrame(data=predictions, columns=['predictions'])
df = pd.concat([dataset_test,det['is_total_final']], axis=1)
df.to_csv("toKamal-1.4.csv")

'''
([$]?[0-9]*[\,]?[0-9]*[\.]?[0-9]+)|(.*((total)(s)?|(amount)).*([$]?([0-9]*[\,]?[0-9]*[\.]?[0-9]+)))
'''
