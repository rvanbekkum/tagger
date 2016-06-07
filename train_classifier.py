from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
import argparse
import numpy as np
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extracts bag of words feature vectors from image training data')
    parser.add_argument('-f', help='directory to feature vectors', required=False, default='features/feature_vectors')
    parser.add_argument('-l', help='directory to labels', required=False, default='features/labels')
    parser.add_argument('-n', help='train for <number> samples', required=False, default=10)
    parser.add_argument('-o', help='filename of trained classifier', required=False, default='model/model.pkl')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    X = []
    y = []
    print "# Loading feature vectors and labels"
    i = 0
    for filename in os.listdir(args.f):
        if i == int(args.n):
            break
        if os.path.isfile(args.l + '/' + filename):
            feature_vector = np.load(args.f + '/' + filename)
            label_vector = np.load(args.l + '/' + filename)
            X.append(feature_vector.tolist())
            y.append(label_vector.tolist())
            i = i + 1
    X = np.matrix(X)
    y = np.matrix(y)

    print "# Training SVM"
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)

    # joblib.dump(clf, args.o)

    sample = '00151757feb88de1777723d170390'
    print(sample)
    vector = np.load('features/feature_vectors/' + sample + '.npy')
    vector = np.matrix(vector)
    prediction = clf.predict(vector)
    # print(prediction)

    print "# Generating labels"

    labels = np.load('labels.npy')
    indexes = [i for i, x in enumerate(prediction[0]) if x == 1]
    for i in indexes:
        print(labels[i])
    #
    # y_vector = np.load('features/labels/001be3fea343798a8833ae9c4d1247.npy')
    # indexes = [i for i, x in enumerate(y_vector) if x == 1]
    # for i in indexes:
    #     print(labels[i])
