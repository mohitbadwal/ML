import pandas as pd
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


data=pd.read_csv(r"C:\\Users\\shubham.kamal\\Desktop\\LITM\\\Not_Success_rows.csv", sep=',',encoding='cp1256')
print('initial shape = ',data.shape)

data = data[data['page_type_final'] == 'remittance']
#data['page_type_final']=data['page_type_final'].astype('category').cat.codes
# is_remit = 0
# is_total = 0
# data['is_remittance'].replace("yes", 1, inplace=True)
# data['is_total'].replace("yes", 1, inplace=True)
# data['is_remittance'].replace("no", 0, inplace=True)
# data['is_total'].replace("no", 0, inplace=True)

print('final shape = ',data.shape)
def cleaning(sentence):
    punctuation_removed = [char for char in sentence if char not in string.punctuation]
    punctuation_removed = [char for char in punctuation_removed if char not in string.digits]
    punctuation_removed = "".join(punctuation_removed)
    l = [word.lower() for word in punctuation_removed.split()]
    return [word for word in l if len(word) > 2]

def cleanandstem(sentence):
    return cleaning(sentence)


tfidf = TfidfVectorizer(tokenizer=cleanandstem, min_df=10,max_df=0.4, stop_words='english')
theString = tfidf.fit_transform(data['row_string'])
combine1 = pd.DataFrame(theString.todense())
combine1.columns = tfidf.get_feature_names()
print(combine1.columns)


# pattern to match totals
pattern = re.compile(
    #"([$]?[0-9]*[\,]?[0-9]*[\.]?[0-9]+)|(.*(total)(s)?|(amount)).*([$]?([0-9]*[\,]?[0-9]*[\.]?[0-9]+))")
    #"(.*[$]*[0-9]*[\,]*[0-9]*[\.]*[0-9]*.*(total)|(amount)|(totals)|(amt)|(paid).*[$]*[0-9]+[\,][0-9]*[\.][0-9]*.*)")
    #"((.*)&((total)|(amount)|(totals)|(amt)|(paid)|(gross))(.*)&([$]*[0-9]+[\,]*[0-9]*[\.]*[0-9]*.*))")
    "(.*(total)|(totals)|(amount)|(amt)|(gross).*[$]*[0-9]+[\,]*[0-9]*[\.]*[0-9]*.*)")
# function for transformation of string to identify the possibility of a total according to the regular expression


def totalFlag(x):
    global pattern

    if pattern.match(str(x).lower()) is not None:
        print(str(x))
        return 1
    else:
        return 0


data['total'] = data['row_string'].apply(totalFlag)

X = data.loc[:, ['total']]
X = pd.concat([combine1.reset_index(drop=True), X.reset_index(drop=True)], axis=1, ignore_index=True)
Y = data.loc[:, 'is_total_final']
validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))