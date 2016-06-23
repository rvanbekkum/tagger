import numpy as np
import os
import feature
import label


def preprocess(sample_size=10, feature_path='data/feature_vectors', label_path='data/labels', delete_old_files=False):

    if delete_old_files:
        print('\n===== REMOVE OLD FILES =====\n')

        output_dirs = ["data/feature_vectors/", "data/labels/", 'model/', 'codebook/']
        for path in output_dirs:
            filelist = [ path + f for f in os.listdir(path) if f.endswith(".npy") or f.endswith('.pkl') ]
            for f in filelist:
                os.remove(f)
                print('Deleted: ' + f)

    print('\n===== FEATURE EXTRACTION =====\n')

    feature.extract(sample_size)

    print('\n===== BINARIZE LABELS =====\n')

    label.binarize(sample_size)

    print('\n===== LOADING FEATURE VECTORS AND LABELS =====\n')

    X = []
    y = []
    i = 0
    for filename in os.listdir(feature_path):
        if i == int(sample_size):
            break
        if os.path.isfile(label_path + '/' + filename):
            feature_vector = np.load(feature_path + '/' + filename)
            label_vector = np.load(label_path + '/' + filename)
            X.append(feature_vector.tolist())
            y.append(label_vector.tolist())
            i = i + 1
    X = np.matrix(X)
    y = np.matrix(y)

    return X, y

def preprocess_test(sample_size=2000, feature_path='data/feature_vectors_test', label_path='data/labels_test'):

    print('\n===== LOADING FEATURE VECTORS AND LABELS =====\n')

    X = []
    y = []
    i = 0
    for filename in os.listdir(feature_path):
        if i == int(sample_size):
            break
        if os.path.isfile(label_path + '/' + filename):
            feature_vector = np.load(feature_path + '/' + filename)
            label_vector = np.load(label_path + '/' + filename)
            X.append(feature_vector.tolist())
            y.append(label_vector.tolist())
            i = i + 1
    X = np.matrix(X)
    y = np.matrix(y)

    return X
