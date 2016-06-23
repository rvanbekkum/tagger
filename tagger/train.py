from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import argparse
import numpy as np
import os
import feature
import label
from preprocess import preprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the classifier')
    parser.add_argument('-f', help='directory to feature vectors', required=False, default='data/feature_vectors')
    parser.add_argument('-l', help='directory to labels', required=False, default='data/labels')
    parser.add_argument('-n', help='train for <number> samples', required=False, default=10000)
    parser.add_argument('-o', help='filename of trained classifier', required=False, default='model/model.pkl')
    return parser.parse_args()

def train(sample_size=10, feature_path='data/feature_vectors', label_path='data/labels', model_path='model/model.pkl'):

    X, y = preprocess(sample_size, feature_path, label_path, delete_old_files=True)

    print('\n===== TRAINING CLASSIFIER =====\n')
    # clf = OneVsRestClassifier(SVC(kernel='linear')).fit(X, y)
    clf = DecisionTreeClassifier().fit(X, y)

    print('\n===== PERSISTING CLASSIFIER =====\n')
    joblib.dump(clf, model_path)

if __name__ == '__main__':
    args = parse_arguments()
    train(args.n, args.f, args.l, args.o)
