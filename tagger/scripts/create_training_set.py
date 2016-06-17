from random import randint
import csv

ORIGINAL_TRAINING_SIZE = 28157620
NEW_TRAINING_SIZE = 10000
TEST_SIZE = 2000
INPUT = '../data/tag-train'
OUTPUT = '../data/training_data.csv'

image_positions = set()

while len(image_positions) < NEW_TRAINING_SIZE + TEST_SIZE:
    image_positions.add(randint(0, ORIGINAL_TRAINING_SIZE - 1))

with open(INPUT, 'r') as csvinput:
    reader = csv.reader(csvinput, delimiter='\t')
    with open(OUTPUT, 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for i, row in enumerate(reader):
            if i in image_positions:
                row.pop(0)
                writer.writerow(row)
                print(row)
