from sklearn.preprocessing import MultiLabelBinarizer
import csv
import numpy as np
import re

IMAGE_DATA = 'image_data.tsv'
OUTPUT = 'features/labels/'
NUM_OF_IMAGES = 10

def tsv_to_list(csvinput):
    image_hashes = []
    labels = []
    with open(csvinput) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            image_hash = row[2]
            tags = row[10].split(',')
            tags_filtered = filter_tags(tags)
            if tags_filtered:
                image_hashes.append(image_hash)
                labels.append(tags_filtered)
    return image_hashes, labels

def filter_tags(tags):
    regex = '(\%[A-Z0-9]{2})+'
    return [tag for tag in tags if not(re.search(regex, tag)) and tag]

if __name__ == '__main__':
    image_hashes, y_labels = tsv_to_list(IMAGE_DATA)
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(y_labels)
    np.save('labels', mlb.classes_)

    y_labeled = zip(image_hashes, y_binary)

    i = 0
    for (hash, y_vector) in y_labeled:
        if i >= NUM_OF_IMAGES:
            break
        np.save(OUTPUT + hash, y_vector)
        print(hash, y_vector)
        i = i + 1
