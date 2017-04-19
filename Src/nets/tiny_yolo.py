import tensorflow.contrib.keras.python.keras as keras
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution2D
from tensorflow.contrib.keras.python.keras.callbacks import *
from tensorflow.contrib.keras.python.keras.optimizers import SGD, Adam
from Src.config import config_yolov2
import numpy as np
from network_skeleton import NetworkSkeleton

class TinyYoloV2(NetworkSkeleton):
    """
    This class handles the building and the loss of the
    tiny yolo v2 network.
    """
    def __init__(self, config):
        """
        Initializes class variables.
        :param config: Contains the networks hyperparameters
        """
        super.__init__(self)
        self.config = config
        self.network = None
        self.loss = None
        self.model = None
        self.input_shape = config.input_shape


    def _build(self, input):
        """
        Builds the tiny yolo v2 network.
        :param input: input image batch to the network
        :return: logits output from network
        """
        self.model = Sequential()
        self.model.add(Lambda(lambda x: x / 127.5 -1.,
                              input_shape=self.input_shape))
        self.model.add(Convolution2D())


        return logits


    def _loss(self):
        """
        Calculates the loss of the network.
        :return: loss
        """
        raise NotImplemented


    def __load_pretrained_network(self):
        """
        Loads the pretrained network's weights
        into the new network.
        :return:
        """