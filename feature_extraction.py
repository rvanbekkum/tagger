import numpy as np
import argparse
import os
from extract_sift import file_to_sift
from sklearn.cluster import KMeans


def get_feature_vector(kmeans, desc):
    labels = kmeans.predict(desc)
    num_centers = len(kmeans.cluster_centers_)
    feature_vector, _ = np.histogram(labels, bins=range(num_centers + 1))
    return feature_vector


def feature_vector_to_file(directory, image_hash, feature_vector):
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory + '/' + image_hash, feature_vector)


def get_sift_descriptors(image_hash, sift_features_dir):
    for filename in os.listdir(sift_features_dir):
        name, file_extension = os.path.splitext(filename)
        if file_extension == '.sift' and image_hash.startswith(name):
            return file_to_sift(filename, sift_features_dir, image_hash)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extracts bag of words feature vectors from image training data')
    parser.add_argument('-m', help='metadata of the training set', required=False, default='image_data.tsv')
    parser.add_argument('-f', help='directory to the SIFT descriptor files', required=False, default='features/sift')
    parser.add_argument('-n', help='extract only for first <number> files', required=False, default=10)
    parser.add_argument('-o', help='output directory of feature vectors', required=False, default='features/feature_vectors')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    number_of_images = 0
    if args.n > 0:
        number_of_images = int(args.n)

    print '# Retrieving SIFT descriptors'
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

    all_sift_descriptors = np.array(all_sift_descriptors)
    num_clusters = int(np.sqrt(all_sift_descriptors.shape[0]))

    print '# Executing K-Means, no. of clusters: {0}'.format(num_clusters)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(all_sift_descriptors)

    print '# Generating feature vectors'
    with open(args.m) as metadata:
        number_processed = 0
        for image_line in metadata:
            number_processed += 1
            if number_processed > number_of_images:
                break
            image_hash = image_line.split()[2]
            (kp, desc) = get_sift_descriptors(image_hash, args.f)
            if desc.shape[0] != 0:
                feature_vector = get_feature_vector(kmeans, desc)
                feature_vector_to_file(args.o, image_hash, feature_vector)
