import numpy as np

class box:
    def __init__(self, classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = 0
        self.probs = np.zeros((classes, 1))

    def process_box(self, config, img_w, img_h, threshold):
        max_indx = np.argmax(self.pred_box.probs)
        max_prob = self.probs[max_indx]
        label = config.labels[max_indx]
        if max_prob > self.threshold:
            left = int((self.x - self.w / 2.) * img_w)
            right = int((self.x + self.w / 2.) * img_w)
            top = int((self.y - self.h / 2.) * img_h)
            bot = int((self.y + self.h / 2.) * img_h)
            if left < 0:  left = 0
            if right > img_w - 1: right = img_w - 1
            if top < 0:   top = 0
            if bot > img_h - 1:   bot = img_h - 1
            mess = '{}'.format(label)
            return (left, right, top, bot, mess, max_indx, max_prob)
        return None



def yolov2_find_boxes(config, predictions):
    raise NotImplemented


def yolo_convert_detections(config, predictions):
    boxes = []
    probs = np.zeros((config.side*config.side*config.num, config.classes))
    for i in range(config.side*config.side):
        row = i / config.side
        col = i % config.side
        for n in range(config.num):
            index = i*config.num+n
            p_index = config.side*config.side*config.classes+i*config.num+n
            scale = config.predictions[p_index]
            box_index = config.side*config.side*(config.classes+config.num) + (i*config.num+n)*4

            new_box = box(config.classes)
            new_box.x = (predictions[box_index + 0] + col) / config.side * config.w
            new_box.y = (predictions[box_index + 1] + row) / config.side * config.h
            new_box.h = pow(predictions[box_index + 2], 2) * config.w
            new_box.w = pow(predictions[box_index + 3], 2) * config.h

            for j in range(config.classes):
                class_index = i*config.classes
                prob = scale*predictions[class_index+j]
                if prob > config.threshold:
                    new_box.probs[j] = prob
                else:
                    new_box.probs[j] = 0
            if config.only_objectness:
                new_box.probs[0] = scale

            boxes.append(new_box)
    return boxes


def prob_compare(boxa, boxb):
    if boxa.probs[boxa.class_num] < boxb.probs[boxb.class_num]:
        return 1
    elif boxa.probs[boxa.class_num] == boxb.probs[boxb.class_num]:
        return 0
    else:
        return -1


def do_nms_sort(boxes, total, classes=20, thresh=0.5):
    for k in range(classes):
        for box in boxes:
            box.class_num = k
        sorted_boxes = sorted(boxes,cmp=prob_compare)
        for i in range(total):
            if sorted_boxes[i].probs[k] == 0:
                continue
            boxa = sorted_boxes[i]
            for j in range(i+1,total):
                boxb = sorted_boxes[j]
                if boxb.probs[k] != 0 and box_iou(boxa,boxb) > thresh:
                    boxb.probs[k] = 0
                    sorted_boxes[j] = boxb
    return sorted_boxes


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    if l1 > l2:
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    if r1 < r2:
        right = r1
    else:
        right = r2
    return right - left


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
         return 0
    area = w*h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w*a.h + b.w*b.h - i
    return u


def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b)
