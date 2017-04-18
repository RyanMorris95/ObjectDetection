from config.config_yolov2 import ConfigYoloV2
from dataset.udacity_data import UdacityData
from nets.tiny_yolo_v2 import TinyYoloV2

# Create Objects
config = ConfigYoloV2()
network = TinyYoloV2(config)
dataset = UdacityData()

def generator():
    raise NotImplemented


def train():


def evaluate():
    raise NotImplemented

def test():
    raise NotImplemented


if __name__ == '__main__':
    train()