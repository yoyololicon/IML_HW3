from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import iris

def laplace_smooth(k, table):
    m, n = table.shape
    for i in range(n):
        total = np.sum(table[:, i])
        for j in range(m):
            table[j, i] = (table[j, i] + k)/(total + k * m)

    return table

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
    print 'decision tree'
    dt = DecisionTreeClassifier(criterion='entropy', presort=True)
    dt.fit(train_data, train_t)
    predict_t = dt.predict(test_data)
    print accuracy_score(test_t, predict_t)

    #k nearest
    print 'knn'
    knn = KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree')
    knn.fit(train_data, train_t)
    predict_t = knn.predict(test_data)
    print accuracy_score(test_t, predict_t)

    #gaussian
    print 'naive bayes using sklearn'
    nb = GaussianNB()
    nb.fit(train_data, train_t)
    predict_t = nb.predict(test_data)
    print accuracy_score(test_t, predict_t)

    #my implementation
    print 'naive bayes from scratch'
    #for continue variable
    table_ctn = np.empty([10, 6, 2])
    table_month = np.zeros([len(month_e.classes_), 6])
    table_day = np.zeros([len(day_e.classes_), 6])
    for i in range(12):
        for j in range(6):
            temp = train_data[np.where(train_t == j), i]
            if i > 3:
                table_ctn[i-2, j, 0] = np.mean(temp)
                table_ctn[i-2, j, 1] = np.std(temp)
            elif i < 2:
                table_ctn[i, j, 0] = np.mean(temp)
                table_ctn[i, j, 1] = np.std(temp)
            elif i == 2:
                for q in range(len(month_e.classes_)):
                    table_month[q, j] = len(temp[np.where(temp == q)])
            elif i == 3:
                for q in range(len(day_e.classes_)):
                    table_day[q, j] = len(temp[np.where(temp == q)])

    table_month = laplace_smooth(3, table_month)
    table_day = laplace_smooth(3, table_day)

    class_prior = np.array([len(np.where(train_t == i)[0])for i in range(6)], dtype=float)
    if np.count_nonzero(class_prior) < 6:
        class_prior = np.squeeze(laplace_smooth(3, class_prior[:, np.newaxis]))
    else:
        class_prior /= np.sum(class_prior)


    predict_t = np.empty(test_t.shape)
    for i in range(len(test_data)):
        candidate = [iris.nb_posterior(class_prior[j],
                                       table_ctn[:, j],
                                       np.concatenate((test_data[i, :2],
                                                       test_data[i, 4:])))
                     * table_month[test_data[i, 2], j]
                     * table_day[test_data[i, 3], j]
                     for j in range(6)]
        predict_t[i] = np.argmax(candidate)
    print accuracy_score(test_t, predict_t)
