import re
import string

import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv(r'C:\Users\mohit.badwal.NOTEBOOK546.000\Downloads\Not_Success_rows_ver.csv',
                      encoding='cp1256')
test_dataset = pd.read_csv(r'D:\backup\PycharmProjects\test\Image Batches-20171017T131547Z-001\Success_rows3.csv',
                           encoding='cp1256')
test_dataset['row_ert'] = test_dataset['row_string']
dataset = dataset[dataset['page_type_final'] == 'remittance'].reset_index()
print(dataset.shape)


def rowIndex(row):
    return row.name


dataset['rowIndex'] = dataset.apply(rowIndex, axis=1)


# countVectorizer = CountVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.5, stop_words='english')
# theString = countVectorizer.fit_transform(dataset['row_string'])
# this function is used to remove punctuations marks and also removes stop words like 'the' , 'a' ,'an'
def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]


'''
.*((inv)|(policy)).*
'''

pattern = re.compile('.*((^|\s)((inv)|(pol)|(amt)|(amo)|((number ){2,}))).*')


def getAlphaNumRatio(x):
    digits = sum(c.isdigit() for c in x)
    letters = sum(c.isalpha() for c in x)
    if letters == 0:
        return 0
    else:
        if (digits / letters) < 0.1:
            return 1
        else:
            return 0


def containsNumber(x):
    i = 0
    for s in x:
        try:
            if 0 >= int(s) <= 9:
                i = i + 1
        except ValueError:
            d = 0
            continue
    return i


def additionalCheck(x):
    s = str(x).strip().split(" ")
    if len(s) > 3:
        return 1
    else:
        return 0


def funcRegEx(e):
    global dataset
    ratio = e['row_numberAlphaRatio']
    x = e['row_string']
    st = str(x).lower().strip()
    i = containsNumber(st)
    if i <= 2:
        if pattern.fullmatch(str(x).lower().strip()):
            if ratio == 1:
                if additionalCheck(str(x).lower().strip()):
                    # if checkHead(dataset, e['rowIndex']-1):
                    # if ratio != e['is_heading']:
                    #     print(ratio, str(x), e['is_heading'])
                    return 1
                    # else:
                    # return 0
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    else:
        return 0


def cleaning_new(sentence):
    fd = '?-+/\\.,'
    punctuation_removed = [char for char in sentence if char not in fd]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return ' '.join(l)


# applying both together
def cleanandstem(sentence):
    return cleaning(sentence)


pat = re.compile('(^|.)*(([a-zA-Z]*[0-9]+).*([$]?[0-9]*[\,]?[0-9]*[\.]?[0-9]+))($|.)*')
pat2 = re.compile('([$]?[0-9]*[\,]?[0-9]*[\.][0-9]+)')


def alphaNumeric(x):
    d = 0
    s = str(x).split(":")
    if len(s) > 1:
        # print(s)
        try:
            d = float(float(s[0]) / float(s[1]))
        except ZeroDivisionError:
            # print("error", s)
            return 0
    else:
        d = float(s[0])
    # print(d)
    if d <= 0.1:
        return 1
    else:
        return 0


number_pattern = re.compile('.*([0-9]{5})+.*')


def checkHead(dataset1, e):
    i = -1
    for r in range(1, 6):
        w = e + r
        s = dataset1.loc[w]['row_string']
        if number_pattern.fullmatch(str(s).lower().strip()) is not None:
            i = 1
            # print(r, str(s).lower())
            break
        else:
            i = 0
    if i == 0:
        # print(dataset1.loc[e]['row_string'])
        return 0
    else:
        return 1


def checkHeading(dataset1):
    i = -1
    for e in range(0, len(dataset1)):
        if dataset1.loc[e]['heading'] == 1:
            for r in range(1, 6):
                w = e + r
                s = dataset1.loc[w]['row_string']

                print(r, w, s)
                if pat.fullmatch(str(s).lower().strip()) is not None:
                    i = 1
                    break
                else:
                    i = 0
            if i == 0:
                # print(dataset1.loc[e]['row_string'])
                dataset1.loc[e, 'heading'] = 0
                print(dataset1.loc[e]['heading'])
    return dataset1


dataset['row_string'] = dataset['row_string'].apply(cleaning_new)
dataset['row_numberAlphaRatio'] = dataset['row_string'].apply(getAlphaNumRatio)
dataset['heading'] = dataset.apply(funcRegEx, axis=1)
test_dataset['row_string'] = test_dataset['row_string'].apply(cleaning_new)
test_dataset['row_numberAlphaRatio'] = test_dataset['row_string'].apply(getAlphaNumRatio)
test_dataset['heading'] = test_dataset.apply(funcRegEx, axis=1)

# print(dataset['row_numberAlphaRatio'].isnull().sum())
# dataset = checkHeading(dataset)
tfidf = CountVectorizer(tokenizer=cleanandstem, min_df=5, stop_words='english',  # ngram_range=(1, 2),
                                                vocabulary=['invoice', 'policy', 'net', 'paid', 'document',

                           'discount', 'inv'])
theString = tfidf.fit_transform(dataset['row_string'])
# a = theString.toarray()
# df_temp = pd.DataFrame(a, columns=tfidf.get_feature_names())
# df_temp = pd.concat([dataset['row_index'],dataset['row_string'], dataset['is_remittance_final'], dataset['is_total_final'],
#                    dataset['is_heading'], df_temp], axis=1)
# df_temp.to_csv("term.csv")
testString = tfidf.transform(test_dataset['row_string'])
combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)
combine2 = pd.DataFrame(testString.todense())
combine2.columns = tfidf.get_feature_names()
X = dataset.loc[:, ['heading',
                    'row_numberAlphaRatio',
                    'row_string'
                    ]]
X = pd.concat([combine1.reset_index(drop=True), X.reset_index(drop=True)], axis=1, ignore_index=True)
Y = dataset.loc[:, 'is_heading']
X = X.iloc[:, :-1]
X1 = test_dataset.loc[:, ['heading',
                          'row_numberAlphaRatio',
                          ]]
X1 = pd.concat([combine2.reset_index(drop=True), X1.reset_index(drop=True)], axis=1, ignore_index=True)
rfc = RandomForestClassifier(n_estimators=200, )
rfc.fit(X, Y)
predictions = rfc.predict(X1)
predictions_prob = rfc.predict_proba(X1)
test_dataset['row_string'] = test_dataset['row_ert']
test_dataset['is_heading'] = pd.DataFrame(data=predictions)
test_dataset.to_csv("test.csv")

'''
Y = dataset.loc[:, 'is_heading']
validation_size = 0.2
seed = 20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

r = X_validation.iloc[:, -1]
X_train = X_train.iloc[:, :-1]

X_validation = X_validation.iloc[:, :-1]


def func(x):
    if x['total'] == 1 and x['pred_proba_0'] < 0.8 and x['pred'] == 0:
        return 1
    return x['pred']


rfc = RandomForestClassifier(n_estimators=200, )
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
predictions_prob = rfc.predict_proba(X_validation)
print(X.columns, rfc.feature_importances_)
pred_prob = pd.DataFrame(data=predictions_prob, columns=[0, 1])
det = pd.DataFrame({"y_val": Y_validation.copy(deep=False).values, "total":
    X_validation.copy(deep=False).iloc[:, -2].values, "pred": predictions, "pred_proba_0": pred_prob[0],
                    "pred_proba_1": pred_prob[1]})
det['pred'] = det.apply(func, axis=1)
a4 = pd.DataFrame(data=predictions, columns=['predictions'])
det.to_csv('det.csv')
print(accuracy_score(det['y_val'], det['pred']))
print(confusion_matrix(det['y_val'], det['pred']))
print(classification_report(det['y_val'], det['pred']))

print(accuracy_score(Y_validation, X_validation.iloc[:, -2].values))
print(confusion_matrix(Y_validation, X_validation.iloc[:, -2].values))
print(classification_report(Y_validation, X_validation.iloc[:, -2].values))

dataset[dataset['is_heading'] != dataset['heading']].loc[:, ['row_string', 'is_heading', 'heading']]
.to_csv('ocr_heading.csv')
'''
