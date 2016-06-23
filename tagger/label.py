from sklearn.preprocessing import MultiLabelBinarizer
import csv
import numpy as np
import re
import os
import joblib


def csv_to_list(n, csvinput='data/training_data.csv', sift_path='data/sift/'):
    image_hashes = []
    labels = []
    with open(csvinput) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == n:
                break
            image_hash = row[0]
            if os.path.isfile(sift_path + image_hash + '.SIFT'):
                tags = row[2].split(',')
                tags_filtered = filter_tags(tags)
                if tags_filtered:
                    image_hashes.append(image_hash)
                    labels.append(tags_filtered)
    return image_hashes, labels

def filter_tags(tags):
    regex = r'(\%[A-Z0-9]{2})+'
    return [tag for tag in tags if re.sub(regex, '', tag)]
    # return [tag for tag in tags if not(re.search(regex, tag)) and tag]

def binarize(n=10, output='data/labels/', csvinput='data/training_data.csv', sift_path='data/sift/', dump_binarizer=False):

    print('Preprocessing labels...')

    sample_size = int(n)
    image_hashes, y_labels = csv_to_list(sample_size, csvinput=csvinput, sift_path=sift_path)
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(y_labels)

    if dump_binarizer:
        joblib.dump(mlb, 'mlb/mlb.pkl')

    np.save(output + 'labels', mlb.classes_)

    y_labeled = zip(image_hashes, y_binary)

    i = 0
    for (hash, y_vector) in y_labeled:
        if i >= n:
            break
        np.save(output + hash, y_vector)
        print(str(i) + ' ' + hash)
        i = i + 1

def binarize_test(n=10, output='data/labels_test/', csvinput='data/test_data.csv', sift_path='data/sift_test/'):

    print('Preprocessing labels...')

    sample_size = int(n)
    image_hashes, y_labels = csv_to_list(sample_size, csvinput=csvinput, sift_path=sift_path)
    mlb = joblib.load('mlb/mlb.pkl')
    y_binary = mlb.transform(y_labels)

    y_labeled = zip(image_hashes, y_binary)

    i = 0
    for (hash, y_vector) in y_labeled:
        if i >= n:
            break
        np.save(output + hash, y_vector)
        print(str(i) + ' ' + hash)
        i = i + 1

if __name__ == '__main__':
    binarize(10000, output='data/labels_test/', csvinput='data/test_data.csv', sift_path='data/sift_test/', dump_binarizer=True)
