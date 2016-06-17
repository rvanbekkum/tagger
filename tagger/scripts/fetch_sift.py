import urllib2, os, re
import csv
from extract_sift import extract_sift

INPUT_TRAIN = '../data/training_data.csv'
INPUT_TEST = '../data/test_data.csv'
OUTPUT = '../data/sift/'

def download_photo(photo_id, user_id):
    url = 'https://www.flickr.com/photos/{}/{}'.format(user_id, photo_id)
    html = urllib2.urlopen(url).read()
    img_url = re.findall(r'(?:https?://)?farm[^":]+_o\.(?:jpg|gif|png)', html)[0].replace('\/', '/')
    filename = OUTPUT + photo_id + '.jpg'

    with open(filename, "wb") as fp:
        fp.write(urllib2.urlopen('https://' + img_url).read())


def fetch_sift(photo_id, user_id):
    filename = OUTPUT + photo_id + '.SIFT'
    if os.path.isfile(filename):
        return
    download_photo(photo_id, user_id)
    extract_sift(filename)
    os.remove(filename)

# Download images and extract SIFT descriptors of the training set
with open(INPUT_TRAIN) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        photo_id = row[0]
        user_id = row[1]
        fetch_sift(photo_id, user_id)