from flask import Flask, request, render_template, jsonify, url_for, send_file, make_response
import cv2
import numpy as np
from flask.ext.mongoengine import MongoEngine
from mongoengine import *
import math
from metadata_extraction import extract_metadata
from profiling import timeit
import StringIO

# Flask application
app = Flask(__name__)
app.debug = True

# Constants
MIN_MATCH_COUNT = 5
MAX_IMAGE_SIZE = 512

# In-memory database
ref_database = []


class ReferenceImage(Document):
    keypoints = ListField(ListField(), required=True)
    descriptors = ListField(ListField(FloatField()), required=True)
    width = IntField(required=True)
    height = IntField(required=True)
    metadata = DynamicField()
    thumbnail = ImageField(size=(800, 600, True), collection_name='reference_thumbnails')

    def to_opencv_description(self):
        ocv_kp = [cv2.KeyPoint(o[0], o[1], o[2]) for o in self.keypoints]
        ocv_des = np.array(self.descriptors, dtype=np.float32)
        return [ocv_kp, ocv_des, self.id, self.width, self.height]


def init_opencv():
    # Initiate SIFT detector
    minHessian = 400
    surf = cv2.SURF(minHessian)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return surf, flann


def init_database():
    app.config["MONGODB_SETTINGS"] = {"DB": "imagematcher"}
    return MongoEngine(app)


def train_matcher(ref_image, descriptors):
    ref_database.append(ref_image)
    matcher.add([descriptors])


def load_db_in_memory():
    ref_database = []
    for o in ReferenceImage.objects:
        ref_image = o.to_opencv_description()
        train_matcher(ref_image, ref_image[1])


@timeit
def open_image(file):
    # convert the data to an array for decoding
    file.seek(0)
    img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 0)

    # rescale to have the largest side at 1024px
    y, x = img.shape
    new_x, new_y = 0, 0

    if x > y:
        new_x = MAX_IMAGE_SIZE
        new_y = y * new_x / x
    else:
        new_y = MAX_IMAGE_SIZE
        new_x = x * new_y / y

    img_resized = cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_AREA)

    return img_resized


def import_image(file):
    img = open_image(file)
    height, width = img.shape

    metadata = extract_metadata(file)

    # find the keypoints and descriptors with SURF
    kp, des = detector.detectAndCompute(img, None)

    # Save the thumbnail
    thumbnail = StringIO.StringIO()
    thumbnail.write(np.array(cv2.imencode(".jpg", img)[1]).tostring())
    thumbnail.seek(0)

    # Store the description of the image in the DB
    converted_kp = [[p.pt[0], p.pt[1], p.size] for p in kp]
    ref_image = ReferenceImage(keypoints=converted_kp,
                               descriptors=des,
                               width=width, height=height,
                               metadata=metadata)
    ref_image.thumbnail.put(thumbnail, content_type='image/jpeg')

    ref_image.save()

    # Keep the important bits in memory
    ocv_ref_image = [kp, des, ref_image.id, width, height]
    train_matcher(ocv_ref_image, des)


def transform_ref_image(mat, w_ref, h_ref):
    pts = np.float32([[0, 0],
                      [0, h_ref - 1],
                      [w_ref - 1, h_ref - 1],
                      [w_ref - 1, 0]])\
            .reshape(-1, 1, 2)
    dst_np = cv2.perspectiveTransform(pts, mat)
    return [o[0].tolist() for o in dst_np]


def normalize(points, w, h):
    scale_point = lambda pt: [pt[0] / w, pt[1] / h]
    return map(scale_point, points)


def score_transformation(mat, w_ref, h_ref):
    # Transform reference image into image space
    dst = transform_ref_image(mat, w_ref, h_ref)
    print dst

    # Compute vectors
    diff_vec = lambda ia, ib:\
        [dst[ib][0] - dst[ia][0], dst[ib][1] - dst[ia][1]]
    lines = [diff_vec(1, 0), diff_vec(2, 1), diff_vec(3, 2), diff_vec(0, 3)]

    # First, make sure the points are ordered clockwise or anti-clockwise
    cross_pdt = lambda u, v: u[0] * v[1] - u[1] * v[0]
    pdts = [cross_pdt(lines[1], lines[0]), cross_pdt(lines[2], lines[1]), cross_pdt(lines[3], lines[2]), cross_pdt(lines[0], lines[3])]

    cur_sign = pdts[0]
    for pdt in pdts:
        if cur_sign * pdt <= 0.0:
            return 0.0, 0.0

    # compute the area of the transformed reference image in the source image
    area = (abs(pdts[0]) + abs(pdts[2])) / 2.0

    # evaluate perspective
    vec_len = lambda v: math.sqrt(v[0] * v[0] + v[1] * v[1])
    vec_lengths = map(vec_len, lines)
    per_1 = vec_lengths[0] / vec_lengths[2]
    per_2 = vec_lengths[1] / vec_lengths[3]
    if per_1 > 1.0:
        per_1 = 1.0 / per_1
    if per_2 > 1.0:
        per_2 = 1.0 / per_2

    if per_1 < 0.5 or per_2 < 0.5:
        return 0.0, 0.0

    # score the transformation: it's the "rectangularity" of the transformed reference image
    sine = [pdts[0] / (vec_lengths[0] * vec_lengths[1]), pdts[1] / (vec_lengths[1] * vec_lengths[2]), pdts[2] / (vec_lengths[2] * vec_lengths[3]), pdts[3] / (vec_lengths[3] * vec_lengths[0])]
    score = sum(sine) / len(sine)

    return score, area


def match_images(kp_img, des_img):
    matches = matcher.knnMatch(des_img, k=2)

    # Filter matches which are more than 3 times further than the min
    #print matches
    #min_dist = min(matches, key=lambda x:x[0].distance)
    #threshold_dist = 3 * min_dist
    #good_matches = filter(lambda x:x[0].distance <= threshold_dist, matches)

    # Go through the list and group by reference image matched
    grouped_matches = {}

    def append_match(m):
        if m.imgIdx in grouped_matches:
            grouped_matches[m.imgIdx].append(m)
        else:
            grouped_matches[m.imgIdx] = [m]

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            append_match(m)
            append_match(n)

    #print grouped_matches

    currentId = None
    currentMat = None
    currentArea = 0.0
    currentScore = 0.0

    # Iterate over each reference image
    for k, ref_matches in grouped_matches.iteritems():
        if len(ref_matches) > MIN_MATCH_COUNT:
            ref_image = ref_database[k]
            mat = find_transformation(kp_img, ref_image, ref_matches)

            if mat is not None:
                score, area = score_transformation(mat, ref_image[3], ref_image[4])

                if score > currentScore:
                    currentMat = mat
                    currentId = ref_image[2]
                    currentArea = area
                    currentScore = score

    #good_matches = []
    #for m, n in matches:
    #    if m.distance < 0.7*n.distance:
    #        good_matches.append(m)
    #        print m.imgIdx

    #if len(good_matches) < MIN_MATCH_COUNT:
    #    return None

    # Get the keypoints from the matches
    #match_kp_img = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    #match_kp_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Find transformation
    #mat, mask = cv2.findHomography(match_kp_img, match_kp_ref, cv2.RANSAC, 5.0)

    return currentId, currentMat, currentArea, currentScore


def find_transformation(kp_img, ref_image, good_matches):
    kp_ref = ref_image[0]

    # Get the keypoints from the matches
    match_kp_img = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    match_kp_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find transformation
    mat, mask = cv2.findHomography(match_kp_img, match_kp_ref, cv2.RANSAC, 5.0)
    return mat


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    msg = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            import_image(file)
            msg = "Upload successful"
        else:
            msg = "Upload failed"

    return render_template('upload.html', message=msg)


@timeit
def process_image(file):
    img = open_image(file)
    img_h, img_w = img.shape

    # find the keypoints and descriptors with SIFT
    kp, des = detector.detectAndCompute(img, None)

    currentId, currentMat, currentArea, currentScore = match_images(kp, des)

    if currentId is None:
        return {"error": "No result"}, 404
    else:
        ref_match = ReferenceImage.objects(id=currentId).first()
        transformed = transform_ref_image(currentMat,
                                          ref_match.width,
                                          ref_match.height)
        transformed_normalized = normalize(transformed, img_w, img_h)
        thumbnail_url = url_for('get_thumbnail', reference_image_id=currentId)

        return {"metadata": ref_match.metadata,
                "area": currentArea,
                "score": currentScore,
                "transformed_normalized": transformed_normalized,
                "thumbnail_url": thumbnail_url}, 200


@app.route('/locate', methods=['GET', 'POST'])
def locate():
    msg = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            result, status_code = process_image(file)
            return make_response(jsonify(result), status_code)
        else:
            msg = "No input file to process"

    return render_template('upload.html', message=msg)


@app.route('/references/', methods=['GET'])
def list_reference_images():
    images = [{"id": str(o.id), "metadata": o.metadata, "thumbnail_url": url_for('get_thumbnail', reference_image_id=o.id)} for o in ReferenceImage.objects]
    response = {"count": len(images), "images": images}
    return jsonify(response)


@app.route('/references/<reference_image_id>/thumbnail', methods=['GET'])
def get_thumbnail(reference_image_id):
    ref_image = ReferenceImage.objects(id=reference_image_id).first()
    thumbnail_file = ref_image.thumbnail
    return send_file(thumbnail_file, mimetype=thumbnail_file.content_type)


@app.route('/match', methods=['POST'])
def match():
    content = request.json
    print content
    return None


@app.route('/clear_db', methods=['GET'])
def clear_db():
    # Empty the database
    for o in ReferenceImage.objects:
        o.delete()

    # Clear the memory cache
    ref_database = []
    detector, matcher = init_opencv()

    return "Database cleared"


# Initialization
detector, matcher = init_opencv()
db = init_database()
load_db_in_memory()

if __name__ == '__main__':
    app.run()
