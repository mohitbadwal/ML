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
def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]


def cleaning_new(sentence):
    fd = '?-+'
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
        return float(d)
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
    for x in li:
        if pattern_number.fullmatch(x) is not None:
            er.append(x)
            i = 1
            if f == -1:
                f = d
        d = d + 1

    if i == 1:
        if convertNumber(er[-1]) == e:
            gh = er[-1]
        li = li[:f]
        li.append(er[-1])
        return ' '.join(li), gh
    else:
        return s, gh


# pattern to match totals
pattern = re.compile(
    "([$]?[0-9]*[\,]?[0-9]*[\.][0-9]+)|(.*((total)(s)?|(amount)).*([$]?([0-9]*[\,]?[0-9]*[\.]?[0-9]+)))")


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
    if gh == 0:
        if pattern.fullmatch(s) is not None:
            return 1
        else:
            # print(str(x))
            return 0
    else:
        return 1


dataset = pd.read_csv(r'D:\backup\PycharmProjects\test\Image Batches-20171017T131547Z-001\Not_Success_rows.csv',
                      encoding='cp1256')
s = ""
is_remit_flag = 0
is_total_flag = 0
# replace yes to 1 and no to 0
# dataset['is_remittance'].replace("yes", 1, inplace=True)
# dataset['is_total'].replace("yes", 1, inplace=True)
# dataset['is_remittance'].replace("no", 0, inplace=True)
# dataset['is_total'].replace("no", 0, inplace=True)


dataset = dataset[dataset['page_type_final'] == 'remittance']
print(dataset.shape)
# countVectorizer = CountVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.5, stop_words='english')
# theString = countVectorizer.fit_transform(dataset['row_string'])
dataset['total'] = dataset.apply(totalFlag, axis=1)
tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=100, stop_words='english')
theString = tfidf.fit_transform(dataset['row_string'])
combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)
X = dataset.loc[:, ['total', 'row_isLastRow']]
X = pd.concat([combine1.reset_index(drop=True), X.reset_index(drop=True)], axis=1, ignore_index=True)
Y = dataset.loc[:, 'is_total_final']
validation_size = 0.2
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print(accuracy_score(dataset['is_total_final'], dataset['total']))
print(confusion_matrix(dataset['is_total_final'], dataset['total']))
print(classification_report(dataset['is_total_final'], dataset['total']))
dataset[dataset['is_total_final'] != dataset['total']].loc[:, ['row_string', 'is_total_final', 'total']] \
    .to_csv('ocr_no_match.csv')

'''
([$]?[0-9]*[\,]?[0-9]*[\.]?[0-9]+)|(.*((total)(s)?|(amount)).*([$]?([0-9]*[\,]?[0-9]*[\.]?[0-9]+)))
'''
