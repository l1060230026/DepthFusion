import numpy as np


class Cameras:
    def __init__(self, cameras, rotation, translation, images_name):
        self.matrix = []
        self.cameras = cameras
        self.length = len(cameras)
        self.images_name = images_name
        self.rotation = rotation
        self.translation = translation
        for i in range(len(cameras)):
            self.matrix.append(np.dot(cameras[i], np.append(rotation[i],
                                                            translation[i].reshape(3, 1), axis=1)))