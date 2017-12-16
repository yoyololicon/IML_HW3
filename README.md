## Environment

* Ubuntu 16.04 LTS
* python2.7.12(using Pycharm 2017.2.3)
* extra module: numpy, sklearn, pandas, scipy

## Usage

Run the following command will display the classification result using dicision tree, KNN and Naive Bayes on two dataset, Iris and Forest fires.

```
python datasets_analyze.py forestfires.csv
```

The Iris will be automatically loaded using sklearn so no need provide csv file.

The output format will like this:

```
----for iris dataset----
decision tree
0.933333333333
[ 1.          0.92307692  0.86956522]
knn
0.955555555556
[ 1.          0.94736842  0.91666667]
naive bayes using sklearn
0.933333333333
[ 1.          0.92307692  0.86956522]
naive bayes from scratch
0.933333333333
[ 1.          0.92307692  0.86956522]
----for forest fires dataset----
decision tree
0.410256410256
knn
0.50641025641
naive bayes using sklearn
0.192307692308
naive bayes from scratch
0.365384615385
```

For Iris dataset, the first output score is accuracy, follow with F-score of each classes.
For Forest fires dataset it will only output the accuracy, cuz the sparsity of target value make it hard to compute F-score.

It will shuffle the data before split the training set and test set so the output value will be vary.
