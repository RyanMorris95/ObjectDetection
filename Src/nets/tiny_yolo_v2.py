import tensorflow.contrib.slim as slim
import tensorflow as tf
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


    def _build(self, input):
        """
        Builds the tiny yolo v2 network.
        :param input: input image batch to the network
        :return: logits output from network
        """
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(input, 16, [3, 3], scope='conv1')
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net)

            net = slim.conv2d(net, 32, [3, 3], scope='conv2')
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net)

            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net)

            net = slim.conv2d(net, 128, [3, 3], scope='conv4')
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net)

            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net)

            net = slim.conv2d(net, 512, [3, 3], scope='conv6')
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net)

            net = slim.conv2d(net, 1024, [3, 3], scope='conv7')
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net)

            net = slim.conv2d(net, 1024, [3, 3], scope='conv8')
            net = slim.batch_norm(net)
            net = slim.conv2d(net, 1024, [3, 3], scope='conv9')
            net = slim.batch_norm(net)

            logits = slim.conv2d(net, 425, [1, 1], activation_fn=None, scope='conv10')

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