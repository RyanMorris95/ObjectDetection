import tensorflow.contrib.slim as slim
import tensorflow as tf
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution2D
from tensorflow.contrib.keras.python.keras.layers import *
from tensorflow.contrib.keras.python.keras.callbacks import *
from tensorflow.contrib.keras.python.keras.optimizers import SGD, Adam
from Src.config import config_yolov2
import Src.Utils.utils as utils
import numpy as np
#from network_skeleton import NetworkSkeleton

class TinyYoloV2:
    """
    This class handles the building and the loss of the
    tiny yolo v2 network.
    """
    def __init__(self, config):
        """
        Initializes class variables.
        :param config: Contains the networks hyperparameters
        """
        #super.__init__(self)
        self.config = config
        self.network = None
        self.loss = None
        self.input_shape = config.input_shape


    def build(self):
        """
        Builds the tiny yolo v2 network.
        :param input: input image batch to the network
        :return: logits output from network
        """
        self.model = Sequential()
        self.model.add(Convolution2D(16, (3, 3), input_shape=(416, 416, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(Convolution2D(32, (3, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(Convolution2D(64, (3, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(Convolution2D(128, (3, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(Convolution2D(256, (3, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        self.model.add(Convolution2D(512, (3, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid'))

        self.model.add(Convolution2D(1024, (3, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(Convolution2D(1024, (3, 3), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())

        self.model.add(Convolution2D(125, (1, 1), activation=None))

        if self.config.optimizer == 'adam':
            opt = Adam()
        elif self.config.optimizer == 'sgd':
            opt = SGD()

        if self.config.loss == 'categorical_crossentropy':
            loss = 'categorical_crossentropy'
        elif self.config.loss == 'yolov2_loss':
            raise NotImplemented

        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        self.model.summary()
        return self.model

    def convertToBoxParams(self, out):
        """
        Convert the final layer features to bounding box parameters.
        :return: box_xy: tensor
                    x, y box predictions adjusted by spatial location in conv layer
                box_wh: tensor
                    w, h box predictions adjusted by anchors and conv spatial resolution
                box_conf: tensor
                    Probability estimate for whether each box contains any object
                box_class_pred : tensor
                    Probability distribution estimate for each box over class labels
        """
        feats = out

        num_anchors = len(self.config.anchors)
        # Reshape to batch, height, width, num_anchors, box_params
        anchors_tensor = K.reshape(K.variable(self.config.anchors), [1, 1, 1, num_anchors, 2])

        conv_dims = K.shape(feats)[1:3]  # assuming channels last
        # In YOLO the height index is the inner most iteration
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

        conv_width_index = K.tile(
            K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))

        feats = K.reshape(
            feats, [-1, conv_dims[0], conv_dims[1], num_anchors, self.config.classes + 5])
        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

        # Static generation of conv_index:
        # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
        # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
        # conv_index = K.variable(
        #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
        # feats = Reshape(
        #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

        box_xy = K.sigmoid(feats[..., :2])
        box_wh = K.exp(feats[..., 2:4])
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.softmax(feats[..., 5:])

        # Adjust preditions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims

        return box_xy, box_wh, box_confidence, box_class_probs


    def save_model(self, name):
        """
        Saves model to path.
        :return:
        """
        path = "C:\ObjectDetection\Data\ModelCheckpoints" + name + ".h5"
        self.model.save(path)

    def save_weights(self, name):
        """
        Saves the model weights to path.
        :param name:
        :return:
        """
        path = "C:\ObjectDetection\Data\ModelCheckpoints" + name + ".h5"
        self.model.save_weights(path)

    def load_weights(self):
        print("About to load weights.")
        utils.load_weights(self.model, 'C:\\ObjectDetection\\Data\\ModelCheckpoints\\tiny-yolo-voc.weights')
        print("Loaded weights.")

    def _load_pretrained_network(self):
        """
        Loads the pretrained network's weights
        into the new network.
        :return:
        """
        raise NotImplemented

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    config = config_yolov2.ConfigYoloV2()
    tinyYoloV2 = TinyYoloV2(config)
    tinyYoloV2.build()
    tinyYoloV2.load_weights()

    imagePath = 'C:\\ObjectDetection\\Data\\test_images\\test1.jpg'
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (416, 416))

    #batch = np.transpose(image,(2,0,1))
    batch = 2*(image/255) - 1
    batch = np.expand_dims(batch, axis=0)
    out = tinyYoloV2.model.predict(batch)
    print ("out:")
    print (out)
    box_xy, box_wh, box_confidence, box_class_probs = tinyYoloV2.convertToBoxParams(out)


    #boxes, _, _ = utils.yolo_eval(out, image.shape)
    #print("Boxes: {}".format(boxes))
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    #ax1.imshow(image)
    #ax2.imshow(utils.draw_box(boxes, plt.imread(imagePath), [[500, 1280], [300, 650]]))
