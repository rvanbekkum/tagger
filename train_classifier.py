import numpy as np
import argparse
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extracts bag of words feature vectors from image training data')
    parser.add_argument('-f', help='directory to feature vectors', required=False, default='features/feature_vectors')
    parser.add_argument('-l', help='directory to labels', required=False, default='features/labels')
    parser.add_argument('-n', help='train for <number> samples', required=False, default=0)
    parser.add_argument('-o', help='filename of trained classifier', required=False, default='trained_svm.pkl')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    X = []
    y = []
    print "# Loading feature vectors and labels"
    for filename in os.listdir(args.f):
        if os.path.isfile(args.l + '/' + filename):
            feature_vector = np.load(args.f + '/' + filename)
            label_vector = np.load(args.l + '/' + filename)
            X.append(feature_vector.tolist())
            y.append(label_vector.tolist())
    X = np.matrix(X)
    y = np.matrix(y)
    #
    print "# Training SVM"
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)

    vector = np.load('features/feature_vectors/0004bb7bbb2676da9b22642423e90.npy')
    vector = np.matrix(vector)
    prediction = clf.predict(vector)
    print(prediction)
    labels = np.load('labels.npy')
    indexes = [i for i, x in enumerate(prediction[0]) if x == 1]
    for i in indexes:
        print(labels[i])

    y_vector = np.load('features/labels/0004bb7bbb2676da9b22642423e90.npy')
    indexes = [i for i, x in enumerate(y_vector) if x == 1]
    for i in indexes:
        print(labels[i])
