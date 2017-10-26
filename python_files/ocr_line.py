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

