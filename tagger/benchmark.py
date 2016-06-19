from preprocess import preprocess
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import numpy as np
import train


X, y = preprocess(1000)

print('\n===== PERFORMING BENCHMARK =====\n')

print('Sample size: {0}...'.format(X.shape[0]))

print('Performing k-fold cross-validation with k = 10...')

kf = KFold(X.shape[0], n_folds=10)

labels = np.load('data/labels/labels.npy')

for train, test in kf:
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    clf = OneVsRestClassifier(SVC(kernel='linear')).fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    prediction = clf.predict(X_test)
    y_array = np.array(y_test).tolist()
    for p, t in zip(prediction, y_array):
        results = [labels[i] for i, x in enumerate(p) if x == 1]
        results_t = [labels[i] for i, x in enumerate(t) if x == 1]
        print(results, results_t)
