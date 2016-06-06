from sklearn.preprocessing import MultiLabelBinarizer
import csv
import numpy as np
import re

IMAGE_DATA = 'image_data.tsv'

def tsv_to_list(csvinput):
    labels = []
    with open(csvinput) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            tags = row[10].split(',')
            tags_filtered = filter_tags(tags)
            labels.append(tags_filtered)
    return labels

def filter_tags(tags):
    regex = '(\%[A-Z0-9]{2})+'
    return [tag for tag in tags if not(re.search(regex, tag))]


y = tsv_to_list(IMAGE_DATA)
y_binary = MultiLabelBinarizer().fit_transform(y)

print(np.shape(y_binary))

np.save('y', y_binary)
