from flask import request
from flask.ext import restful
from models.referenceimage import ReferenceImage


class MetadataAPI(restful.Resource):
    def get(self, ref_image_id):
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        return ref_image.metadata

    def put(self, ref_image_id):
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        json = request.get_json(force=True, silent=True, cache=False)
        print json
        if ref_image is not None:
            if json is not None:
                ref_image.metadata = json
                ref_image.save()
                return ref_image.to_simple_object()
            else:
                restful.abort(500, message="Invalid metadata")
        else:
            restful.abort(404, message="Invalid image")
