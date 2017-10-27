import string

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


# applying both together
def cleanandstem(sentence):
    return cleaning(sentence)


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
tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=50,max_df=0.4, stop_words='english')
theString = tfidf.fit_transform(dataset['row_string'])
combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)
X = dataset.loc[:, ['check_noOfPages']]
X = pd.concat([combine1.reset_index(drop=True), X.reset_index(drop=True)], axis=1, ignore_index=True)
Y = dataset.loc[:, 'is_total_final']
validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X, Y)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



'''

([$]?[0-9]*[\,]?[0-9]*[\.]?[0-9]+)|((total)(s)?|(amount)).*([$]?([0-9]*[\,]?[0-9]*[\.]?[0-9]+))

'''