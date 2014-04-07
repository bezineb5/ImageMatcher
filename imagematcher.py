from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from flask.ext.mongoengine import MongoEngine
from mongoengine import *
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.debug = True

MIN_MATCH_COUNT = 5
MAX_IMAGE_SIZE = 1024

class ReferenceImage(Document):
    name = StringField(required=True)
    coordinates = GeoPointField()
    keypoints = ListField(ListField())
    descriptors = ListField(ListField(FloatField()))
    width = IntField(required=True)
    height = IntField(required=True)

    def to_opencv_description(self):
        ocv_kp = [cv2.KeyPoint(o[0], o[1], o[2]) for o in self.keypoints]
        ocv_des = np.array(self.descriptors,dtype=np.float32)
        return [ocv_kp, ocv_des, self.id, self.width, self.height]

def init_opencv():
    # Initiate SIFT detector
    minHessian = 400
    surf = cv2.SURF(minHessian)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return surf, flann

def init_database():
    app.config["MONGODB_SETTINGS"] = {"DB": "imagematcher"}
    return MongoEngine(app)

def load_db_in_memory():
    #for o in ReferenceImage.objects:
    #    o.delete()
    return [o.to_opencv_description() for o in ReferenceImage.objects]

def open_image(file):
    # convert the data to an array for decoding
    img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 0)

    # rescale to have the largest side at 1024px
    x,y = img.shape
    new_x, new_y = 0, 0
    if x > y:
        new_x = MAX_IMAGE_SIZE
        new_y = y * new_x / x
    else:
        new_y = MAX_IMAGE_SIZE
        new_x = x * new_y / y

    img_resized = cv2.resize(img, (new_x, new_x), interpolation=cv2.INTER_AREA)

    return img_resized

def import_image(file):
    name = secure_filename(file.filename)
    img = open_image(file)
    w, h = img.shape

    # find the keypoints and descriptors with SIFT
    kp, des = detector.detectAndCompute(img, None)
    converted_kp = [ [p.pt[0], p.pt[1], p.size] for p in kp]
    refImage = ReferenceImage(name=name, coordinates=None, keypoints=converted_kp, descriptors=des, width=w, height=h).save()

    ocv_ref_image = [kp, des, refImage.id, w, h]
    ref_database.append(ocv_ref_image)

def score_transformation(mat, h_img, w_img, h_ref, w_ref):
    pts = np.float32([ [0,0],[0,h_ref-1],[w_ref-1,h_ref-1],[w_ref-1,0] ]).reshape(-1,1,2)
    dst_np = cv2.perspectiveTransform(pts, mat)
    dst = [o[0].tolist() for o in dst_np]

    print dst

    # Compute vectors
    diff_vec = lambda ia, ib: [dst[ib][0] - dst[ia][0], dst[ib][1] - dst[ia][1]]
    lines = [ diff_vec(1, 0), diff_vec(2, 1), diff_vec(3, 2), diff_vec(0, 3) ]

    # First, make sure the points are ordered clockwise or anti-clockwise
    cross_pdt = lambda u, v: u[0]*v[1] - u[1]*v[0]
    pdts = [cross_pdt(lines[1], lines[0]), cross_pdt(lines[2], lines[1]), cross_pdt(lines[3], lines[2]), cross_pdt(lines[0], lines[3])]

    cur_sign = pdts[0]
    for pdt in pdts:
        if cur_sign*pdt <= 0.0:
            return 0.0

    # compute the area of the transformed reference image in the source image
    area = (abs(pdts[0]) + abs(pdts[2])) / 2.0

    # evaluate perspective
    vec_len = lambda v: v[0]*v[0] + v[1]*v[1]
    per_1 = vec_len(lines[0])/vec_len(lines[2])
    per_2 = vec_len(lines[1])/vec_len(lines[3])
    if per_1 > 1.0:
        per_1 = 1.0/per_1
    if per_2 > 1.0:
        per_2 = 1.0/per_2

    if per_1 < 0.5 or per_2 < 0.5:
        return 0.0

    return area


def match_images(kp_img, des_img, kp_ref, des_ref):
    matches = flann.knnMatch(des_img, des_ref, k=2)
    
    # Filter matches which are more than 3 times further than the min
    #print matches
    #min_dist = min(matches, key=lambda x:x[0].distance)
    #threshold_dist = 3 * min_dist
    #good_matches = filter(lambda x:x[0].distance <= threshold_dist, matches)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    if len(good_matches) < MIN_MATCH_COUNT:
        return None, 0, 0

    # Get the keypoints from the matches
    match_kp_img = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    match_kp_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Find transformation
    mat, mask = cv2.findHomography(match_kp_img, match_kp_ref, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    print len(matchesMask)

    return mat, len(good_matches), len(matchesMask)

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

@app.route('/locate', methods=['GET', 'POST'])
def locate():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = open_image(file)
            img_h, img_w = img.shape

            # find the keypoints and descriptors with SIFT
            kp, des = detector.detectAndCompute(img, None)

            currentMat = None
            currentArea = 0.1
            currentId = None

            for ref in ref_database:
                mat, numTotal, numMask = match_images(kp, des, ref[0], ref[1])

                if mat is not None:
                    print "total: " + str(numTotal) + " mask:" + str(numMask)
                    ratio = float(numMask)/float(numTotal)
                    print ratio

                    area = score_transformation(mat, img_h, img_w, ref[3], ref[4])

                    if area > currentArea:
                        currentMat = mat
                        currentId = ref[2]
                        currentArea = area

            if currentId is None:
                return jsonify({"error": "No result"})
            else:
                ref_match = ReferenceImage.objects(id=currentId).first()

                return jsonify({"name": ref_match.name, "area": currentArea})

    return render_template('upload.html', message=None)

@app.route('/match', methods=['POST'])
def match():
    content = request.json
    print content
    return None

detector, flann = init_opencv()
db = init_database()
ref_database = load_db_in_memory()

if __name__ == '__main__':
    app.run()
