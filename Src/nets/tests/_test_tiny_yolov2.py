import Src.nets.tiny_yolo_v2 as model
import Src.config.config_yolov2 as config
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow.contrib.keras.api.keras as keras

"""
Overfits the tiny_yolov2 to one image of the dataset.
"""