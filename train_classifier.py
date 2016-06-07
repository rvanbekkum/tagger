import numpy as np
import argparse
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

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
        feature_vector = np.load(args.f + '/' + filename)
        label_vector = np.load(args.l + '/' + filename)
        X.append(feature_vector)
        y.append(label_vector)
    print "# Training SVM"
    #clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
    #clf.predict(np.load('features/feature_vectors/001bcce2f7f5846fbdc36583bedb6.npy'))