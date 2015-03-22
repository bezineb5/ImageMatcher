from flask import Flask, request, render_template
from flask.ext import restful
from flask.ext.mongoengine import MongoEngine
import os
from resources.thumbnail import ThumbnailAPI
from resources.metadata import MetadataAPI
from resources.referenceimage import ReferenceImageAPI
from resources.music import MusicAPI
from resources.search import SearchAPI
from resources.referenceimagelist import ReferenceImageListAPI
from engine.matcher import import_image, init_opencv, load_db_in_memory


# Flask application
app = Flask(__name__)
app.debug = True

# RESTful API
api = restful.Api(app)

api.add_resource(ReferenceImageListAPI,
                 '/api/1.0/references/',
                 endpoint='reference_image_list')
api.add_resource(ReferenceImageAPI,
                 '/api/1.0/references/<string:ref_image_id>',
                 endpoint='reference_image')
api.add_resource(MetadataAPI,
                 '/api/1.0/references/<string:ref_image_id>/metadata',
                 endpoint='metadata')
api.add_resource(ThumbnailAPI,
                 '/api/1.0/references/<string:ref_image_id>/thumbnail',
                 endpoint='thumbnail')
api.add_resource(MusicAPI,
                 '/api/1.0/references/<string:ref_image_id>/music',
                 endpoint='music')
api.add_resource(SearchAPI,
                 '/api/1.0/search',
                 endpoint='search')


def init_database():
    app.config["MONGODB_SETTINGS"] = {
        'db': "imagematcher",
        'host': os.environ.get('OPENSHIFT_MONGODB_DB_HOST', ''),
        'port': int(os.environ.get('OPENSHIFT_MONGODB_DB_PORT', 0)),
        'username': os.environ.get('OPENSHIFT_MONGODB_DB_USERNAME', ''),
        'password': os.environ.get('OPENSHIFT_MONGODB_DB_PASSWORD', '')
    }
    return MongoEngine(app)


@app.route('/')
def main_page():
    return render_template('index_polymer.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    msg = None

    if request.method == 'POST':
        image_file = request.files['file']
        if image_file:
            import_image(image_file)
            msg = "Upload successful"
        else:
            msg = "Upload failed"

    return render_template('upload.html', message=msg)


# Initialization
#init_opencv()
init_database()
load_db_in_memory()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=52300)
