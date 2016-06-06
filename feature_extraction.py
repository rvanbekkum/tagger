import numpy as np
import argparse
import os
from extract_sift import file_to_sift

def generate_codewords_dictionary(sift):
    print '# Generating codewords dictionary'


def get_feature_vector(codebook):
    print '# Generating feature vector'


def extract_training_features(training_data_filename):
    print '# Extracting feature vectors for training'


def get_sift_descriptors(image_hash, sift_features_dir):
    for filename in os.listdir(sift_features_dir):
        if image_hash.startswith(os.path.splitext(filename)):
            return file_to_sift(filename, sift_features_dir, image_hash)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extracts bag of words feature vectors from image training data')
    parser.add_argument('-m', help='metadata of the training set', required=False, default='image_data.tsv')
    parser.add_argument('-f', help='directory to the SIFT descriptor files', required=False, default='features/sift')
    parser.add_argument('-n', help='extract only for first <number> files', required=False, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()

    #test = np.array([[0,1],[0,2]])
    #print np.concatenate((test,[[4,4],[3,3],[6,6]]))

    number_of_images = 0
    if args.n > 0:
        number_of_images = int(args.n)

    all_sift_descriptors = []
    with open(args.m) as metadata:
        number_processed = 0
        for image_line in metadata:
            number_processed += 1
            if number_processed > number_of_images:
                break
            image_hash = image_line.split()[2]
            (kp, desc) = get_sift_descriptors(image_hash, args.f)
            for descriptor in desc:
                all_sift_descriptors.append(descriptor)
    np.array(all_sift_descriptors)
    # do k-means