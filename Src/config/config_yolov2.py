import numpy as np

class ConfigYoloV2:
    def __init__(self):
        self.name = "Tiny Yolo V2"
        self.anchors = np.array([[1.08, 1.19],
                                 [3.42, 4.41],
                                 [6.63, 11.38],
                                 [9.42, 5.11],
                                 [16.62, 10.52]])
        self.bias_match = 1
        self.classes = 3
        self.coords = 4
        self.num = 5
        self.softmax = 1
        self.jitter = 0.2
        self.rescore = 1

        self.object_scale = 5
        self.noobject_scale = 1
        self.class_scale = 1
        self.coord_scale = 1

        self.absolute = 1
        self.thresh = .6
        self.random = 1

        self.checkpoint_path = '/home/ryan/PycharmProjects/yolo/Data/ModelCheckpoints'

        self.input_size = (418, 418)