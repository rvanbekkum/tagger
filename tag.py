from sklearn.externals import joblib
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extracts bag of words feature vectors from image training data')
    parser.add_argument('-f', help='directory to feature vectors', required=False, default='features/feature_vectors')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

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
