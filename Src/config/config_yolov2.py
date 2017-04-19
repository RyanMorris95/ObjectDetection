import numpy as np

class ConfigYoloV2:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 200
        self.data_augmentation = True
        self.loss = 'categorical_crossentropy'
        self.name = "Tiny Yolo V2"
        self.anchors = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38],
                        [9.42, 5.11], [16.62, 10.52]]
        self.bias_match = 1
        self.classes = 20
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
        self.thresh = .5
        self.random = 1

        self.checkpoint_path = '/home/ryan/PycharmProjects/yolo/Data/ModelCheckpoints'

        self.input_shape = [418, 418, 3]

        self.optimizer = 'adam'
