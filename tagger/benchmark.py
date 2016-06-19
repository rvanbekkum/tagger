from preprocess import preprocess
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import train

def is_similar(predictions, truth):
    num_of_similar = len(filter(lambda x: x in predictions, truth))
    if num_of_similar < len(truth) / 2.0:
        return False
    else:
        return True

X, y = preprocess(10000)

print('\n===== PERFORMING BENCHMARK =====\n')

print('Sample size: {0}...'.format(X.shape[0]))

print('Performing k-fold cross-validation with k = 10...')

kf = KFold(X.shape[0], n_folds=10)

labels = np.load('data/labels/labels.npy')

for train, test in kf:
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    clf = OneVsRestClassifier(SVC(kernel='linear')).fit(X_train, y_train)
    # clf = DecisionTreeClassifier().fit(X_train, y_train)
    # clf = RandomForestClassifier().fit(X_train, y_train)
    # clf = KNeighborsClassifier().fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    prediction = clf.predict(X_test)
    y_array = np.array(y_test).tolist()

    num_of_correct_predictions = 0.0

    for p, t in zip(prediction, y_array):
        pred_tags = [labels[i] for i, x in enumerate(p) if x == 1]
        true_tags = [labels[i] for i, x in enumerate(t) if x == 1]
        # print(pred_tags, true_tags)
        if is_similar(pred_tags, true_tags):
            num_of_correct_predictions += 1

    print('Accuracy: ' + str(num_of_correct_predictions / len(prediction)))
