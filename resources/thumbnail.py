from flask import send_file
from flask.ext import restful
from models.referenceimage import ReferenceImage


class ThumbnailAPI(restful.Resource):
    def get(self, ref_image_id):
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        thumbnail_file = ref_image.thumbnail
        return send_file(thumbnail_file, mimetype=thumbnail_file.content_type)
