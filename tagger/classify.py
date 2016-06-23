from sklearn.externals import joblib
import numpy as np
import argparse
import os.path
import sys
from scripts.extract_sift import extract_sift
from feature import get_feature_vector

def predict_label(sift_desc, codebook, clf):
    feature_vector = get_feature_vector(codebook, sift_desc)
    return clf.predict(feature_vector)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict tags for a new image.')
    parser.add_argument('-f', help='path to image to predict tags for', required=True)
    parser.add_argument('-m', help='path to trained classifier', required=False, default='model/model.pkl')
    parser.add_argument('-c', help='path to codebook', required=False, default='codebook/codebook.pkl')
    parser.add_argument('-l', help='path to label-mapping', required=False, default='data/labels/labels.npy')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    if not os.path.isfile(args.f):
        sys.exit('Image not found: {}'.format(args.f))

    print('\n===== LOADING CLASSIFIER =====\n')

    clf = joblib.load(args.m)
    print('Successfully loaded classifier')

    print('\n===== LOADING CODEBOOK =====\n')

    codebook = joblib.load(args.c)
    print('Successfully loaded codebook')

    print('\n===== EXTRACTING SIFT DESCRIPTORS =====\n')

    kp, desc = extract_sift(args.f)
    print('Successfully extracted SIFT descriptors')

    print('\n===== PREDICTING LABELS =====\n')

    prediction = predict_label(desc, codebook, clf)
    labels = np.load(args.l)
    pred_tags = [labels[i] for i, x in enumerate(prediction[0]) if x == 1]

    print(pred_tags)
