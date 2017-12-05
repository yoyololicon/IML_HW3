from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
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

    #gaussian
    nb = GaussianNB()
    nb.fit(train_data, train_t)
    predict_t = nb.predict(test_data)
    print accuracy_score(test_t, predict_t)

    #my implementation

    #for continue variable
    table_ctn = np.empty([12, 6, 2])
    for i in range(12):
        for j in range(6):
            temp = train_data[np.where(train_t == j), i]
            table_ctn[i, j, 0] = np.mean(temp)
            table_ctn[i, j, 1] = np.std(temp)

    class_prior = np.array([len(np.where(train_t == i)[0])for i in range(6)], dtype=float)
    #print np.min(table_ctn[:, :, 0])
    if np.count_nonzero(class_prior) < 6:
        class_prior = np.squeeze(laplace_smooth(1, class_prior[:, np.newaxis]))
    else:
        class_prior /= np.sum(class_prior)

    #print class_prior - nb.class_prior_
    #print nb.sigma_ - np.power(table_ctn[:, :, 1].T, 2)
    #print nb.theta_  - table_ctn[:, :, 0].T

    predict_t2 = np.empty(test_t.shape)
    for i in range(len(test_data)):
        candidate = [iris.nb_posterior(class_prior[j], table_ctn[:, j], test_data[i])
                     for j in range(6)]
        predict_t2[i] = np.argmax(candidate)
    print accuracy_score(test_t, predict_t2)

if __name__ == '__main__':
    run_forestfires()
