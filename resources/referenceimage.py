from flask.ext import restful
from models.referenceimage import ReferenceImage


class ReferenceImageAPI(restful.Resource):
    def get(self, ref_image_id):
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        return ref_image.to_simple_object()

    def delete(self, ref_image_id):
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        if ref_image is not None:
            ref_image.delete_attachments()
            ref_image.delete()
        pass
