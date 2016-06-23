import csv


TRAINING_DATA = '../data/training_data.csv'

labels = set()

with open(TRAINING_DATA) as file:
    reader = csv.reader(file)
    for row in reader:
        tags = row[2].split(',')
        for tag in tags:
            labels.add(tag)

print(len(labels))
