from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from scipy.stats import norm

class_dtb = [1./3, 1./3, 1./3]

def nb_posterior(prior, posterior, x):
    result = prior
    for p, xx in zip(posterior, x):
        result *= norm.pdf(xx, loc=p[0], scale=p[1])
    return result

def run_iris():
    iris = load_iris()
    train_data, test_data, train_t, test_t = train_test_split(iris.data, iris.target, test_size=0.3)

    #decision tree
    print 'decision tree'
    dt = DecisionTreeClassifier(criterion='entropy', presort=True)
    dt.fit(train_data, train_t)
    predict_t = dt.predict(test_data)
    print accuracy_score(test_t, predict_t)
    print f1_score(test_t, predict_t, average=None)

    #k nearest
    print 'knn'
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    knn.fit(train_data, train_t)
    predict_t = knn.predict(test_data)
    print accuracy_score(test_t, predict_t)
    print f1_score(test_t, predict_t, average=None)

    print 'naive bayes using sklearn'
    #std gaussian
    nb = GaussianNB()
    nb.fit(train_data, train_t)
    predict_t = nb.predict(test_data)
    print accuracy_score(test_t, predict_t)
    print f1_score(test_t, predict_t, average=None)

    print 'naive bayes from scratch'
    #my implementation
    table = np.empty([4, 3, 2])
    for i in range(4):
        for j in range(3):
            temp = train_data[np.where(train_t == j), i]
            table[i, j, 0] = np.mean(temp)
            table[i, j, 1] = np.std(temp)
    #print table

    predict_t = np.empty(test_t.shape)
    for i in range(len(test_data)):
        candidate = [nb_posterior(class_dtb[j], table[:, j], test_data[i]) for j in range(3)]
        predict_t[i] = np.argmax(candidate)
    print accuracy_score(test_t, predict_t)
    print f1_score(test_t, predict_t, average=None)