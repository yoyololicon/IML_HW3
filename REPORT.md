# Homework 3 for Introduction to Machine Learning

## 開發環境和語言

詳細資訊可查看[readme](README.md)。

## Iris Dataset

### Dicision Tree

使用了sikit learn來實做決策樹。
從sklearn官方的[文件](http://scikit-learn.org/stable/modules/tree.html)來看其決策樹的建構方式不完全是ID3，算是其延伸。
為了和課堂教的ID3做最大的連結，將分割的標準定為entropy。

### KNN 
一樣用了sklearn來實做KNN，樹的建構方式為KD tree，k=5。

### Naive Bayes

sklearn也有提供naive bayes的函式，且由於資料都是連續的，可以直接餵進去不用做轉換。
不過我還是自己做了一個，方式也很簡單，只要將training data裡，分別屬於三個類別的資料分開，去算他們各feature的平均和標準差，也就是屬於該class的條件機率。
預測時只要選擇三種類別，其各feature條件機率相乘數值中最大的類別就好。
在測試時和sklearn的輸出結果近乎相同。

### 測試結果

實際測試其實三種的表現方式都差不多，難分軒輊，但是隨著資料分割的不同，有時候會有其中一種表現特別差的情形。
例如以下dicision tree的效果就較差：

```
#accuracy
#[class1_F-score    class2_F-score    class3_F-score]

decision tree
0.844444444444
[ 1.    0.72  0.8 ]
knn
0.977777777778
[ 1.          0.96        0.97142857]
naive bayes using sklearn
0.933333333333
[ 1.          0.88888889  0.90909091]
naive bayes from scratch
0.933333333333
[ 1.          0.88888889  0.90909091]
```

也有的時候是KNN表現較差：

```
0.933333333333
[ 1.          0.89655172  0.92682927]
knn
0.866666666667
[ 1.          0.8125      0.84210526]
naive bayes using sklearn
0.911111111111
[ 1.          0.875       0.89473684]
naive bayes from scratch
0.911111111111
[ 1.          0.875       0.89473684]
```

甚至只有NB表現好的時候：

```
decision tree
0.844444444444
[ 1.          0.8         0.78787879]
knn
0.866666666667
[ 1.          0.83333333  0.8125    ]
naive bayes using sklearn
0.933333333333
[ 1.          0.90909091  0.91428571]
naive bayes from scratch
0.933333333333
[ 1.          0.90909091  0.91428571]
```

總和來說，表現最穩定的是naive bayes，其他兩種似乎都蠻吃訓練資料的正確性。

個人分析，決策樹比較適合用在category類型的feature，用連續的資料雖然可以劃分區間來轉成category，但不能完全代表原資料的型態。

KNN則是先架構出training data的空間分佈，再去從test data的位置去找附近的點，適合用在連續的資料。
但這樣會有一個問題就是沒考慮到training data各類別的數量和分佈密度，有可能輸入資料屬於A但A分佈稀少且較散，就有可能misclassify。
而Naive Bayes用機率的方式可以表達出這些問題。
不過從結果來看KNN比決策樹穩一些。

## Forest Fires

###資料處理

由於資料有category的feature，需要轉成數字代表。屬於這樣的feature有幾月跟星期幾。
而target value 'area'是連續的，反而要轉成category，因為我們的演算法是分類而非回歸。
共分成：t=0, 0<t<=1, 1<t<=10, 10<t<=100, 100<t<=1000, 1000<t六個類別。

###Decision Tree

和Iris同。

###KNN

和Iris同，只是k=10。

###Naive Bayes

基本上和Iris相同，但是category featrue要獨立出來算。
而且有些feature算出的機率可能是0，也要用laplace smooth處理，不然會預測不出training set沒有的組合。

而在這個dataset有一個有趣的現象，就是area > 1000的資料只有一個，也就是class 6只會有一個instance。
由於所有條件機率一開始數值都是0，不管是該instance有沒有在training set或沒有在training set，其標準差都會是0，用scipy.norm算其機率時會算不出來。
所以只好多設一個標準差為0時，任何輸入數值只要不等於平均其機率為0的條件。

### 測試結果

以下為某次測試結果，數值為accuracy：

```
decision tree
0.333333333333
knn
0.480769230769
naive bayes using sklearn
0.217948717949
naive bayes from scratch
0.435897435897
```

KNN和Naive bayes表現不相上下，決策樹則較差。
個人覺得可能是feature和Iris相比多很多，而其中可能有些feature並不是很重要，決策樹如果用這些feature做決策會誤導結果。
用sklearn做的Naive Bayes其結果差不意外，因為輸入資料有兩個feature被轉換成數字，其數值大小並沒有意義。

比較令我驚訝的是KNN，Naive Bayes有特別對category feature處理，直接餵資料的KNN卻可以和Naive Bayes一樣準，甚至更好。

我想到的解釋可能是，這個dataset其各feature之間並不是完全獨立的，而Naive Bayes預設feature都彼此獨立，所以可能會高估或低估一個feature的機率。
因為Naive Bayes如果要區分出各class，只能畫出線性的、圓形或橢圓的分隔線，而KNN的分隔線就沒有這樣的限制。