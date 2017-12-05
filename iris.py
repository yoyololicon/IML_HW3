from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def run_iris():
    iris = load_iris()
    train_data, test_data, train_t, test_t = train_test_split(iris.data, iris.target, test_size=0.3)

    #decision tree
    dt = DecisionTreeClassifier(criterion='entropy', presort=True)
    dt.fit(train_data, train_t)
    predict_t = dt.predict(test_data)
    print accuracy_score(test_t, predict_t)
    print f1_score(test_t, predict_t, average=None)

    #k nearest
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
    knn.fit(train_data, train_t)
    predict_t = knn.predict(test_data)
    print accuracy_score(test_t, predict_t)
    print f1_score(test_t, predict_t, average=None)

    #gaussian
    nb = GaussianNB()
    nb.fit(train_data, train_t)
    predict_t = nb.predict(test_data)
    print accuracy_score(test_t, predict_t)
    print f1_score(test_t, predict_t, average=None)