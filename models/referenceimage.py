from flask import url_for
from mongoengine import ListField, IntField, Document, DynamicField, ImageField, FileField
import cv2
import numpy as np


class ReferenceImage(Document):
    keypoints = ListField(ListField(), required=True)
    # SURF: descriptors = ListField(ListField(FloatField()), required=True)
    # ORB
    descriptors = ListField(ListField(IntField()), required=True)
    width = IntField(required=True)
    height = IntField(required=True)
    metadata = DynamicField()
    thumbnail = ImageField(size=(800, 600, True),
                           collection_name='reference_thumbnails')
    music_attachment = FileField(collection_name='music_attachments')

    def to_opencv_description(self):
        ocv_kp = [cv2.KeyPoint(o[0], o[1], o[2]) for o in self.keypoints]
        # SURF: ocv_des = np.array(self.descriptors, dtype=np.float32)
        # For BRISK descriptors
        ocv_des = np.array(self.descriptors, dtype=np.uint8)
        return [ocv_kp, ocv_des, self.id, self.width, self.height]

    def to_simple_object(self):
        ref = {"id": str(self.id),
               "metadata": self.metadata,
               "thumbnail_url": url_for('thumbnail',
                                        ref_image_id=self.id),
               }

        music_file = self.music_attachment
        if music_file is not None and music_file.get() is not None:
            ref["music_url"] = url_for('music',
                                       ref_image_id=self.id)

        return ref

    def delete_attachments(self):
        if self.thumbnail is not None:
            self.thumbnail.delete()
            self.thumbnail = None
        music_file = self.music_attachment
        if music_file is not None and music_file.get() is not None:
            self.music_attachment = None
            music_file.delete()
        self.save()
