from sklearn.preprocessing import MultiLabelBinarizer
import csv
import numpy as np
import re

IMAGE_DATA = 'data/training_data.csv'
OUTPUT = 'data/labels/'

def csv_to_list(n, csvinput=IMAGE_DATA):
    image_hashes = []
    labels = []
    with open(csvinput) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == n:
                break
            image_hash = row[0]
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

def binarize(n=10):

    print('Preprocessing labels...')

    sample_size = int(n)
    image_hashes, y_labels = csv_to_list(sample_size)
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(y_labels)
    np.save(OUTPUT + 'labels', mlb.classes_)

    y_labeled = zip(image_hashes, y_binary)

    i = 0
    for (hash, y_vector) in y_labeled:
        if i >= n:
            break
        np.save(OUTPUT + hash, y_vector)
        print(str(i) + ' ' + hash)
        i = i + 1

binarize()
