from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
import argparse
import numpy as np
import os
import feature
import label


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extracts bag of words feature vectors from image training data')
    parser.add_argument('-f', help='directory to feature vectors', required=False, default='data/feature_vectors')
    parser.add_argument('-l', help='directory to labels', required=False, default='data/labels')
    parser.add_argument('-n', help='train for <number> samples', required=False, default=10)
    parser.add_argument('-o', help='filename of trained classifier', required=False, default='model/model.pkl')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    print('\n===== REMOVE OLD FILES =====\n')

    output_dirs = ["./data/feature_vectors/", "./data/labels/", 'model/']
    for path in output_dirs:
        filelist = [ path + f for f in os.listdir(path) if f.endswith(".npy") or f.endswith('.pkl') ]
        for f in filelist:
            os.remove(f)
            print('Deleted: ' + f)

    print('\n===== FEATURE EXTRACTION =====\n')

    feature.extract(args.n)

    print('\n===== BINARIZE LABELS =====\n')

    label.binarize(args.n)

    print('\n===== LOADING FEATURE VECTORS AND LABELS =====\n')

    X = []
    y = []
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

    print X
    print y

    print('\n===== TRAINING CLASSIFIER =====\n')
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)

    print('\n===== PERSISTING CLASSIFIER =====\n')
    joblib.dump(clf, args.o)
