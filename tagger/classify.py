from sklearn.externals import joblib
import numpy as np


if __name__ == '__main__':

    print('\n===== LOADING CLASSIFIER =====\n')

    clf = joblib.load('model/model.pkl')

    print('\n===== PREDICTING LABELS =====\n')

    sample = '0004bb7bbb2676da9b22642423e90'
    # print(sample)
    vector = np.load('data/feature_vectors/' + sample + '.npy')
    vector = np.matrix(vector)
    prediction = clf.predict(vector)

    labels = np.load('data/labels/labels.npy')
    indexes = [i for i, x in enumerate(prediction[0]) if x == 1]
    for i in indexes:
        print(labels[i])

    y_vector = np.load('data/labels/0004bb7bbb2676da9b22642423e90.npy')
    indexes = [i for i, x in enumerate(y_vector) if x == 1]
    for i in indexes:
        print(labels[i])
