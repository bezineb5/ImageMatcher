from flask import request, send_file
from flask.ext import restful
from models.referenceimage import ReferenceImage


class MusicAPI(restful.Resource):
    def get(self, ref_image_id):
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        music_file = ref_image.music_attachment
        if music_file is not None and music_file.get() is not None:
            return send_file(music_file, mimetype=music_file.content_type)
        else:
            restful.abort(404, message="No music attached")

    def post(self, ref_image_id):
        self.delete(ref_image_id)
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        music_file = request.files['music']
        if music_file:
            import_music(ref_image, music_file)
            ref_image.save()
            return ref_image.to_simple_object()
        else:
            restful.abort(404, message="Upload failed")

    def put(self, ref_image_id):
        self.post(ref_image_id)

    def delete(self, ref_image_id):
        ref_image = ReferenceImage.objects(id=ref_image_id).first()
        if ref_image is not None:
            music_file = ref_image.music_attachment
            if music_file is not None and music_file.get() is not None:
                ref_image.music_attachment = None
                music_file.delete()
                self.save()
        pass


def import_music(ref_image, file):
    if file is not None:
        music_file = ref_image.music_attachment
        if music_file is not None and music_file.get() is not None:
            music_file.delete()

        music_file.put(file, content_type='audio/mpeg')
