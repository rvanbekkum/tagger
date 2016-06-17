from random import randint
import csv

ORIGINAL_TRAINING_SIZE = 28157620
NEW_TRAINING_SIZE = 10000
TEST_SIZE = 2000
INPUT = '../data/tag-train'
OUTPUT_TRAIN = '../data/training_data.csv'
OUTPUT_TEST = '../data/test_data.csv'

image_training_positions = set()
image_test_positions = set()

while len(image_training_positions) < NEW_TRAINING_SIZE:
    image_training_positions.add(randint(0, ORIGINAL_TRAINING_SIZE - 1))

while len(image_test_positions) < TEST_SIZE:
    pos = randint(0, ORIGINAL_TRAINING_SIZE - 1)
    if pos not in image_training_positions:
        image_test_positions.add(pos)

with open(INPUT, 'r') as csvinput:
    reader = csv.reader(csvinput, delimiter='\t')
    with open(OUTPUT_TRAIN, 'w') as csv_train:
        train_writer = csv.writer(csv_train)
        with open(OUTPUT_TEST, 'w') as csv_test:
            test_writer = csv.writer(csv_test)
            for i, row in enumerate(reader):
                if i in image_training_positions:
                    row.pop(0)
                    train_writer.writerow(row)
                    print('TRAINING DATA: ', row)
                if i in image_test_positions:
                    row.pop(0)
                    test_writer.writerow(row)
                    print('TESTING DATA: ', row)
