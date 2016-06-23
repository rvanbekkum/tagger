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

def accuracy_at_k(predictions, truth, k=99999):
    return any(i in truth for i in predictions[:k])

X, y = preproces_test(2000)

print('\n===== PERFORMING EVALUATION =====\n')

print('Sample size: {0}...'.format(X.shape[0]))

labels = np.load('data/labels/labels.npy')

clf = joblib.load('model/model.pkl')
prediction = clf.predict(X)

num_of_correct_predictions = 0.0

for p, t in zip(prediction, y_array):
    pred_tags = [labels[i] for i, x in enumerate(p) if x == 1]
    true_tags = [labels[i] for i, x in enumerate(t) if x == 1]
    # print(pred_tags, true_tags)
    if accuracy_at_k(pred_tags, true_tags):
        num_of_correct_predictions += 1

print('Accuracy: ' + str(num_of_correct_predictions / len(prediction)))
