from flask import request
from flask.ext import restful
from models.referenceimage import ReferenceImage
from common import dbcache
from engine.matcher import import_image, init_opencv
from resources.music import import_music


class ReferenceImageListAPI(restful.Resource):
    def get(self):
        images = map(lambda o: o.to_simple_object(), ReferenceImage.objects)
        return {"count": len(images), "images": images}

    def post(self):
        image_file = request.files['image']
        if image_file:
            ref_image = import_image(image_file)
            music_file = request.files['music']
            if music_file:
                import_music(ref_image, music_file)
                ref_image.save()

            return ref_image.to_simple_object()
        else:
            restful.abort(404, message="Upload failed")

    def delete(self):
        # Empty the database
        for o in ReferenceImage.objects:
            o.delete()

        # Clear the memory cache
        dbcache.clear()
        init_opencv()

        return {"success": True, "message": "Database cleared"}
