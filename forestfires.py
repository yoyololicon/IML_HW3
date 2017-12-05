from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def target_convert(t):
    for i in range(len(t)):
        if t[i] > 0 and t[i] <= 1:
            t[i] = 1
        elif t[i] > 1 and t[i] <= 10:
            t[i] = 2
        elif t[i] > 10 and t[i] <= 100:
            t[i] = 3
        elif t[i] > 100 and t[i] <= 1000:
            t[i] = 4
        elif t[i] > 1000:
            t[i] = 5
    return t

def run_forestfires():
    ff = pd.read_csv('forestfires.csv').values
    [data, target] = np.split(ff, [12], axis=1)
    target = np.squeeze(target_convert(target)).astype(int)

    month_e = LabelEncoder()
    day_e = LabelEncoder()
    data[:, 2] = month_e.fit_transform(data[:, 2])
    data[:, 3] = day_e.fit_transform(data[:, 3])

    train_data, test_data, train_t, test_t = train_test_split(data, target, test_size=0.3)

    #decision tree
    dt = DecisionTreeClassifier(criterion='entropy', presort=True)
    dt.fit(train_data, train_t)
    predict_t = dt.predict(test_data)
    print accuracy_score(test_t, predict_t)

    #k nearest
    knn = KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree')
    knn.fit(train_data, train_t)
    predict_t = knn.predict(test_data)
    print accuracy_score(test_t, predict_t)

    #gaussian
    nb = GaussianNB()
    nb.fit(train_data, train_t)
    predict_t = nb.predict(test_data)
    print accuracy_score(test_t, predict_t)
