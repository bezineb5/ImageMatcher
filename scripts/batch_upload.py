from os import listdir
from os.path import isfile, join, basename, splitext
import sys
import requests

SERVER_URL = "http://127.0.0.1:5000/"


def upload_files(base_url, files):
    url = base_url + 'upload'
    for f in files:
        file_upload = {'file': (basename(f), open(f, 'rb'), 'image/*')}
        requests.post(url, files=file_upload)


def image_from_dir(path):
    only_files = [f for f in listdir(path) if isfile(join(path, f))]
    only_images = [join(path, f) for f in only_files if splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')]
    return only_images


def main(argv):
    if len(argv) > 0:
        path = argv[0]
        files = image_from_dir(path)
        upload_files(SERVER_URL, files)
    else:
        print "Missing argument: path to the directory containing the images"


if __name__ == "__main__":
    main(sys.argv[1:])
