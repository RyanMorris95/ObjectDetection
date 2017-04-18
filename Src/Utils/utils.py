import numpy as np

def yolov2_find_boxes(config, net_out, threshold):
    sqrt = config.sqrt + 1
    C, B, S = config.classes, config.num, config.side
    boxes = []
    SS = S * S  # number of grid cells
    prob_size = SS * C  # class probabilities
    conf_size = SS * B  # Confidences for each grid cell

    probs = np.ascontiguousarray(net_out[0 : prob_size]).reshape([SS, C])

