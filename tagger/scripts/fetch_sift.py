import urllib2, os, re
import csv
from extract_sift import extract_sift

INPUT_TRAIN = '../data/training_data.csv'
INPUT_TEST = '../data/test_data.csv'
OUTPUT = '../data/sift/'

def download_photo(photo_id, user_id):
    url = 'https://www.flickr.com/photos/{}/{}'.format(user_id, photo_id)
    try:
        html = urllib2.urlopen(url).read()
    except urllib2.HTTPError, err:
        return
    img_urls = re.findall(r'(?:https?://)?farm[^":]+_o\.(?:jpg|gif|png)', html)
    if len(img_urls) > 0:
        img_url = img_urls[0].replace('\/', '/')
        filename = OUTPUT + photo_id + '.jpg'

        with open(filename, "wb") as fp:
            fp.write(urllib2.urlopen('https://' + img_url).read())


def fetch_sift(photo_id, user_id):
    filename = OUTPUT + photo_id
    if os.path.isfile(filename + '.SIFT'):
        return
    download_photo(photo_id, user_id)
    if os.path.isfile(filename + '.jpg'):
        extract_sift(filename + '.jpg')
        os.remove(filename + '.jpg')

# Download images and extract SIFT descriptors of the training set
with open(INPUT_TRAIN) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    num = 1
    for row in reader:
        photo_id = row[0]
        user_id = row[1]
        print 'Fetching file #{}'.format(num)
        fetch_sift(photo_id, user_id)
        num += 1
        if num == 5001:
            break