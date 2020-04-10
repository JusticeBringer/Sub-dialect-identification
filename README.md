# Romanian sub-dialect identification
Discriminate between the Moldavian and the Romanian dialects across different text genres (news versus tweets)\
**(Project details can also be found on the Kaggle competition _[here](https://www.kaggle.com/c/ml-2020-unibuc-3/data)_)**


## Task
Participants have to train a model on tweets. Therefore, participants have to build a model for a in-genre binary classification by dialect task, in which a classification model is required to discriminate between the Moldavian (label 0) and the Romanian (label 1) dialects.

The training data is composed of 7757 samples. The validation set is composed of 2656 samples. All samples are preprocessed in order to replace named entities with a special tag: $NE$.

**Note that the tweets are encrypted. You may or not decrypt the texts for solving this project.**

## File description 

* train_samples.txt - the training data samples (one sample per row)
* train_labels.txt - the training labels (one label per row)
* validation_samples.txt - the validation data samples (one sample per row)
* validation_labels.txt - the validation labels (one label per row)
* test_samples.txt - the test data samples (one sample per row)
* sample_submission.txt - a sample submission file in the correct format

## Data format
You can download the data for this project **_[here](https://github.com/JusticeBringer/Sub-dialect-identification/tree/master/data)_**

### Samples File
The data samples are provided in the following format based on TAB separated values:

```
112752	sAFW K#xk}t fH@ae m&Xd >h& @# l@Rd}a @Hc liT ehAr@m Xgmz !}a }eAr@m Be g@@m efH RB(D Ehk&
107227	X;d:N qnwB Acke@m m*g lvc& ggcp ht*A mat; }:@ HA&@m HA@e hZ Er#@m
101685	#fEw w!ygF dDB XwfE| HrWe@mH
```

Each line represents a data sample where:

* The first column shows the ID of the data sample.
* The second column is the actual data sample.

### Labels file 
The labels are provided in the following format based on TAB separated values:

```
1    1
2    0
```
Each line represents a label associated to a data sample where:

* The first column shows the ID of the data sample.
* The second column is the actual label.

## Evaluation 

The evaluation measure is the macro F1 score computed on the test set. The macro F1 score is given by the mean of the F1 scores computed for each class. The F1 score is given by:
F1=2 * (P⋅R)/(P+R),
where P is the precision and R is the recall.

You can try your submissions **_[here](https://www.kaggle.com/c/ml-2020-unibuc-3/submit)_**

# How I approached this project and solved it with ~66% accuracy score

For solving the sub-dialect identification I tried all the following models:

```
1. Logistic Regression 
2. Naïve Bayes
3. SVM
4. Linear Regression
5. LSTM
6. LSTM + CNN
7. GRU cells
```

All the 4 first models had similar workflow:

```
1. Load the data from the .txt files
2. Apply the TF-IDF technique
(3.*) Sometimes the data was normalized and standardised
4. Fit and predict

```

## Results on the validation data

### 1. Logistic Regression

```
Logistic Regression accuracy score -> 0.6295180722891566

                Confusion matrix
                [[792 509]
                 [475 880]]

                Classification report

              precision    recall  f1-score   support

           0       0.63      0.61      0.62      1301
           1       0.63      0.65      0.64      1355

    accuracy                           0.63      2656
   macro avg       0.63      0.63      0.63      2656
weighted avg       0.63      0.63      0.63      2656

Logistic Regression F1 score: 0.629360764766923
```

### 2. Naïve Bayes 

```
Naïve Bayes accuracy score -> 65.54969879518072

                  Confusion matrix
                  [[760 541]
                   [374 981]]

                  Classification report

              precision    recall  f1-score   support

           0       0.67      0.58      0.62      1301
           1       0.64      0.72      0.68      1355

    accuracy                           0.66      2656
   macro avg       0.66      0.65      0.65      2656
weighted avg       0.66      0.66      0.65      2656

Naïve Bayes F1 score: 0.6536820451582341
```

### 3. SVM

```
SVM accuracy score -> 0.6321536144578314

                    Confusion matrix
                    [[780 521]
                     [456 899]]

                    Classification report
              
              precision    recall  f1-score   support

           0       0.63      0.60      0.61      1301
           1       0.63      0.66      0.65      1355

    accuracy                           0.63      2656
   macro avg       0.63      0.63      0.63      2656
weighted avg       0.63      0.63      0.63      2656

SVM F1 score: 0.6317494637382586
```

### 4. Linear Regression

```
Linear Regression accuracy score -> 0.5835843373493976

                      Confusion matrix
                      [[755 546]
                       [560 795]]

                      Classification report

              precision    recall  f1-score   support

           0       0.57      0.58      0.58      1301
           1       0.59      0.59      0.59      1355

    accuracy                           0.58      2656
   macro avg       0.58      0.58      0.58      2656
weighted avg       0.58      0.58      0.58      2656

Linear Regression F1 score: 0.583617401506497
```

The last 3 models had also similar workflow:

```
1. Load the data from .txt files
2. Assign each crypted word its frequency number
3. Fit and predict
```

### 5. LSTM

```
LSTM accuracy score -> 0.6788403614457831

                        Confusion matrix
                        [[813 488]
                         [365 990]]

                        Classification report

              precision    recall  f1-score   support

           0       0.69      0.62      0.66      1301
           1       0.67      0.73      0.70      1355

    accuracy                           0.68      2656
   macro avg       0.68      0.68      0.68      2656
weighted avg       0.68      0.68      0.68      2656

LSTM F1 score: 0.6778447812774927
```

### 6. LSTM + CNN

```
LSTM with CNN accuracy score -> 0.6125753012048193

                          Confusion matrix
                          [[796 505]
                           [524 831]]

                           Classification report

              precision    recall  f1-score   support

           0       0.60      0.61      0.61      1301
           1       0.62      0.61      0.62      1355

    accuracy                           0.61      2656
   macro avg       0.61      0.61      0.61      2656
weighted avg       0.61      0.61      0.61      2656

LSTM with CNN F1 score: 0.6126118294013413
```

### 7. GRU cells

```
GRU cells accuracy score -> 0.5101656626506024

                          Confusion matrix
                          [[   0 1301]
                           [   0 1355]]
 
                          Classification report
 
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      1301
           1       0.51      1.00      0.68      1355

    accuracy                           0.51      2656
   macro avg       0.26      0.50      0.34      2656
weighted avg       0.26      0.51      0.34      2656

GRU cells F1 score: 0.3446893407586967
```

## Conclusion on validation data

* Best accuracy score was LSTM with ~67%
* Second best accuracy score was Naïve Bayes with ~65%
* Best F1 score was LSTM with ~67%
* Second best F1 score was Naïve Bayes with ~65%

# Programming language

* Python

# Author

* Arghire Gabriel (me)
