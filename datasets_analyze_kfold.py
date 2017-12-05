from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import numpy as np

def main():
    iris_d, iris_t = load_iris(True)
    data, target = shuffle(iris_d, iris_t)
    splits = 3
    kf = KFold(n_splits=splits)

    acc = [[], [], []]
    f1 = [[], [], []]

    for train_index, test_index in kf.split(data):
        #decision tree
        dt = DecisionTreeClassifier(criterion='entropy', presort=True)
        dt.fit(data[train_index], target[train_index])
        predict_t = dt.predict(data[test_index])
        acc[0].append(accuracy_score(target[test_index], predict_t))
        f1[0].append(f1_score(target[test_index], predict_t, average=None))

        #k nearest
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
        knn.fit(data[train_index], target[train_index])
        predict_t = knn.predict(data[test_index])
        acc[1].append(accuracy_score(target[test_index], predict_t))
        f1[1].append(f1_score(target[test_index], predict_t, average=None))

        #gaussian
        nb = GaussianNB()
        nb.fit(data[train_index], target[train_index])
        predict_t = nb.predict(data[test_index])
        acc[2].append(accuracy_score(target[test_index], predict_t))
        f1[2].append(f1_score(target[test_index], predict_t, average=None))

    print 'average accuracy of decision tree is', sum(acc[0])/splits
    print 'average accuracy of KNN is', sum(acc[1])/splits
    print 'average accuracy of naive bayes is', sum(acc[2])/splits

    f1 = np.array(f1)
    print 'average F-score of decision tree is'
    print np.mean(f1[0, :, 0]), np.mean(f1[0, :, 1]), np.mean(f1[0, :, 2])
    print 'average F-score of KNN is'
    print np.mean(f1[1, :, 0]), np.mean(f1[1, :, 1]), np.mean(f1[1, :, 2])
    print 'average F-score of naive bayes is'
    print np.mean(f1[2, :, 0]), np.mean(f1[2, :, 1]), np.mean(f1[2, :, 2])

if __name__ == '__main__':
    main()