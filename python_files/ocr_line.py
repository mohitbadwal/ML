import string
import sys
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

if len(sys.argv) == 3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
else:
    print("Format python filename train_dataset_path test_dataset_path")
    sys.exit(1)


def minOps(A, B):
    m = len(A)
    n = len(B)
    s = max(n, m)
    if n != m:
        f = 0
        s = max(n, m)
        s2 = min(n, m)
        if s == m:
            f = 1
        for i in range(s2, s):
            if f == 0:
                A = A + '+'
            else:
                B = B + '+'

    count = 0
    for der in range(s):
        if A[der] != B[der]:
            count += 1
    return count


def cleaning(sentence):
    fd = '?-+:)(;,*\'°—'
    punctuation_removed = [char for char in sentence if char not in fd]
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    # punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return ' '.join([word for word in l if len(word) > 2])


def cleaning_new(sentence):
    fd = '?-+:)(;,*\'°—'
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
pattern_number = re.compile('([$]?[0-9]*[\,]?[0-9]*[\.]?[0-9]+)')
regx = re.compile('(?<![\d.])'
                  '(?!\d*\.\d*\.)'  # excludes certain string as not being numbers
                  '((\d|\.\d)([\d.])*?)'  # the only matching  group
                  '([0\.]*)'
                  '(?![\d.])')


def convertNumber(s):
    # print(s)
    d = ''
    for i in str(s):
        if i != ',' and i != '$':
            d = d + i
    if len(d) > 0:
        return (float(d))
    else:
        return -1


def convertNumber_new(s):
    # print(s)
    d = ''
    for i in str(s):
        if i != ',' and i != '$' and i != '.':
            d = d + i
    if len(d) > 0:
        return (float(d))
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
    # print(er, er1, i)
    if i == 1:
        f = 0
        for x in er:
            # print("printing", x, e)
            if convertNumber(x) >= 0:  # int(float(e)):
                gh = 5
                # li = li[:f]
                # print("here", e)
                li.append(x)
                f = 1
                break
        if f == 0:
            li.append(er[-1])
        return ' '.join(li), gh
    else:
        return s, gh


def secondChance(s, e):
    e = cleaning(str(e))
    # length_integer = len(e)
    print(s, e)
    li = str(s).split(" ")
    i = 0
    gh = 0
    d = 0
    f = -1
    er = []
    er1 = []
    for x in li:
        number = re.findall(r'\d+', x)
        if len(number) > 0:
            er.append(number[0])
            er1.append(d)
            i = 1
            if f == -1:
                f = d
        d = d + 1

    er1.sort(reverse=True)
    # print(len(er1),li)
    for j in range(0, len(er1)):
        li.pop(er1[j])
    # print(er, er1, i)

    faltu = int((regx.sub('\\1', e)))
    length_integer = len(str(faltu)) - 1
    dope = 2
    if length_integer <= 3:
        dope = 1
    if i == 1:
        f = 0
        for x in er:
            # print("printing", x, e)
            # if (int(float(e))-(10**(length_integer-1))) <= convertNumber(x) <= (int(float(e))-(10**(length_integer-1))):
            faltu2 = int(convertNumber_new(regx.sub('\\1', x)))
            print(faltu, faltu2)
            if 0 <= minOps(str(faltu), str(faltu2)) <= dope:
                gh = 5
                # li = li[:f]
                # print("here", e)
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
    # print(s)
    s, gh = isNumber(s, e)
    # print(s,gh)
    # if gh == 0:
    if pattern.fullmatch(s) is not None:
        if re.search('(gross)', s) is None and re.search('(render)', s) is None \
                and re.search('(comm(\s)?%)|(com(\s)?%)|(comm(ission)?)', s) is None:
            if gh != 0:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        # print(str(x))
        # give a second chance
        # s = str(eddd).lower().strip()
        # s, gh_second = secondChance(s, e)
        # if gh_second != 0:
        #    return 1
        return 0
        # else:
        #  return 1


def afterPred(x):
    if x['pred'] == 0:  # or (x['pred'] == 1 and x['total_regex'] == 0):
        eddd = str(x['str'])
        e = x['check_amount']
        eddd = cleaning(eddd)
        s = str(eddd).lower().strip()
        # if re.search('(statement)',s) is None:
        s, gh_second = secondChance(s, e)
        if gh_second != 0:
            print(s, gh_second, 1)
            return 1
        else:
            #   if x['pred'] == 1:
            #       return x['pred']
            #  else:
            return 0
    return x['pred']


dataset = pd.read_csv(train_file, encoding='cp1256')
dataset_test = pd.read_csv(test_file, encoding='cp1256')
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
                # print(x.loc[i - 1]['row_isLastRow'])
    return x


# dataset = dataset[dataset['page_type_final'] == 'remittance'].reset_index()
print(dataset.shape)
print(dataset_test.shape)

# countVectorizer = CountVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.5, stop_words='english')
# theString = countVectorizer.fit_transform(dataset['row_string'])

dataset['rows'] = dataset['page_noOfRows'] - dataset['row_rowNumber']
dataset['total_regex'] = dataset.apply(totalFlag, axis=1)
dataset = last(dataset)

dataset_test['rows'] = dataset_test['page_noOfRows'] - dataset_test['row_rowNumber']
dataset_test['total_regex'] = dataset_test.apply(totalFlag, axis=1)
dataset_test = last(dataset_test)

tfidf = TfidfVectorizer(tokenizer=cleanandstem, stop_words='english', vocabulary=
{'totals',
 'total',
 'grand'
 })
theString = tfidf.fit_transform(dataset['row_string'])
# from sklearn.externals import joblib
# joblib.dump(tfidf,"tfidf_ocr_total.pkl")
# tfidf = joblib.load("tfidf_ocr_total.pkl")

combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)

theTestString = tfidf.transform(dataset_test['row_string'])
# from sklearn.externals import joblib
# joblib.dump(tfidf,"tfidf_ocr_total.pkl")
# tfidf = joblib.load("tfidf_ocr_total.pkl")

combine2 = pd.DataFrame(theTestString.todense())
combine2.columns = tfidf.get_feature_names()
dataset['totals'] = combine1['totals']
dataset['total'] = combine1['total']
dataset['grand'] = combine1['grand']
dataset_test['totals'] = combine2['totals']
dataset_test['total'] = combine2['total']
dataset_test['grand'] = combine2['grand']
# da = dataset.columns
# da_test = dataset_test.columns
# print(dataset.shape,combine1.shape)
# dataset = pd.DataFrame(pd.concat([combine1.reset_index(drop=True), dataset.reset_index(drop=True)], axis=1,
#                                 ignore_index=True),columns=da)
# dataset = dataset.reset_index(drop=True).append(combine1,ignore_index=True)

# dataset.to_csv("wwwww.csv")
# dataset_test = pd.DataFrame(pd.concat([combine2.reset_index(drop=True), dataset_test.reset_index(drop=True)], axis=1,
#                                 ignore_index=True),columns=da_test)
# dataset_test=dataset_test.reset_index(drop=True).append(combine2.reset_index(drop=True),ignore_index=True)
X = dataset.loc[:, [  # 'row_distanceFromTop',
    # 'row_isLastRow',
    # 'total_regex',

    'rows',
    'row_string',
    'check_checkAmount']]
X = pd.concat([combine1.reset_index(drop=True), X.reset_index(drop=True)], axis=1, ignore_index=True)
Y = dataset.loc[:, 'is_total_final']

X_test = dataset_test.loc[:, [  # 'row_distanceFromTop',
    # 'row_isLastRow',
    # 'total_regex',
    'rows',
    'row_string',
    'check_checkAmount']]
X_test = pd.concat([combine2.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1, ignore_index=True)

X = X.iloc[:, :-2]
er = X_test.iloc[:, -2]
ch = X_test.iloc[:, -1]
X_test = X_test.iloc[:, :-2]


def func(x):
    if x['total'] == 1 and x['pred_proba_0'] < 0.88 and x['pred'] == 0:
        return 1
    return x['pred']


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, Y)
print(rfc.feature_importances_)
predictions = rfc.predict(X_test)
predictions_prob = rfc.predict_proba(X_test)
pred_prob = pd.DataFrame(data=predictions_prob, columns=[0, 1])
det = pd.DataFrame({"str": er.values, "check_amount": ch.values, "total_regex":
    dataset_test['total_regex'], "pred": predictions, "pred_proba_0": pred_prob[0],
                    "pred_proba_1": pred_prob[1], "is_total": dataset_test['is_total_final_original']})

# det['pred'] = det.apply(func, axis=1)
det['pred'] = det.apply(afterPred, axis=1)
dataset_test['is_total_final'] = det['pred']
'''
data_group = dataset_test.groupby(by=['check_checkNumber', 'check_checkAmount', 'page_pageNumber'])
dataset_new = pd.DataFrame()
for name, group in data_group:
    group.reset_index(inplace=True)
    group['is_total_new'] = 0
    for g_pos in range(len(group) - 1, -1, -1):
        # print(g_pos)
        if group.iloc[g_pos]['is_total_final'] == 1:
            group.loc[g_pos, 'is_total_new'] = 1
            # print(g_pos)
            break
    dataset_new = pd.concat([dataset_new, group.reset_index()], axis=0, ignore_index=True)
# dataset_test.to_csv(test_file)
dataset_new.to_csv(test_file)
# '''

dataset_test.to_csv(test_file)
# det.to_csv(r"D:\LITM_Data\Modspace\CSVs\det1.csv")
# a4 = pd.DataFrame(data=predictions, columns=['predictions'])
# df = pd.concat([er.reset_index(), det['y_val'], det['pred']], axis=1)
# df.to_csv("wer.csv")


'''
([$]?[0-9]*[\,]?[0-9]*[\.]?[0-9]+)|(.*((total)(s)?|(amount)).*([$]?([0-9]*[\,]?[0-9]*[\.]?[0-9]+)))
'''
