import urllib2, os, re
from extract_sift import extract_sift


def download_photo(user_id, photo_id):
    url = 'https://www.flickr.com/photos/{}/{}'.format(user_id, photo_id)
    html = urllib2.urlopen(url).read()
    img_url = re.findall(r'(?:https?://)?farm[^":]+_o\.jpg', html)[0].replace('\/', '/')

    with open(photo_id + '.jpg', "wb") as fp:
        fp.write(urllib2.urlopen('https://' + img_url).read())

    return photo_id + '.jpg'


def fetch_sift(user_id, photo_id):
    filename = download_photo(user_id, photo_id)
    extract_sift(filename)
    os.remove(filename)