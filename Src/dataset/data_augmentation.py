import numpy as np
import cv2
import skimage.transform
import tensorflow as tf

class DataAugmentation:
    def __init__(self, width, height, newWidth, newHeight):
        self.width = width
        self.height = height
        self.newWidth = newWidth
        self.newHeight = newHeight


    def batch_iou(self, boxes, box):
        """Compute the Intersection-Over-Union of a batch of boxes with another
        box.

        Args:
          box1: 2D array of [cx, cy, width, height].
          box2: a single array of [cx, cy, width, height]
        Returns:
          ious: array of a float number in range [0, 1].
        """
        lr = np.maximum(
            np.minimum(boxes[:, 0] + 0.5 * boxes[:, 2], box[0] + 0.5 * box[2]) - \
            np.maximum(boxes[:, 0] - 0.5 * boxes[:, 2], box[0] - 0.5 * box[2]),
            0
        )
        tb = np.maximum(
            np.minimum(boxes[:, 1] + 0.5 * boxes[:, 3], box[1] + 0.5 * box[3]) - \
            np.maximum(boxes[:, 1] - 0.5 * boxes[:, 3], box[1] - 0.5 * box[3]),
            0
        )
        inter = lr * tb
        union = boxes[:, 2] * boxes[:, 3] + box[2] * box[3] - inter
        return inter / union

    def trans_image(self, image, bb_boxes_f, trans_range):
        # Translation augmentation
        bb_boxes_f = bb_boxes_f.copy(deep=True)

        if np.random.randint(2) == 1:
            tr_x = trans_range * np.random.uniform() - trans_range / 2
            tr_y = trans_range * np.random.uniform() - trans_range / 2
            bb_boxes_f['xmin'] = bb_boxes_f['xmin'] + tr_x
            bb_boxes_f['xmax'] = bb_boxes_f['xmax'] + tr_x
            bb_boxes_f['ymin'] = bb_boxes_f['ymin'] + tr_y
            bb_boxes_f['ymax'] = bb_boxes_f['ymax'] + tr_y
        else:
            tr_x = trans_range * np.random.uniform() + trans_range / 2
            tr_y = trans_range * np.random.uniform() + trans_range / 2
            bb_boxes_f['xmin'] = bb_boxes_f['xmin'] + tr_x
            bb_boxes_f['xmax'] = bb_boxes_f['xmax'] + tr_x
            bb_boxes_f['ymin'] = bb_boxes_f['ymin'] + tr_y
            bb_boxes_f['ymax'] = bb_boxes_f['ymax'] + tr_y

        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        rows, cols, channels = image.shape

        image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

        return image_tr, bb_boxes_f

    def random_flip(self, image, bb_boxes):
        if np.random.randint(2) == 1:
            image = cv2.flip(image, 1)  # vertical flip
            xmin = bb_boxes['xmin']
            width = bb_boxes['xmax'] - bb_boxes['xmin']
            bb_boxes['xmin'] = np.round(image.shape[1] - bb_boxes['xmin'])
            xmax = bb_boxes['xmax']
            bb_boxes['xmax'] = np.round(image.shape[1] - bb_boxes['xmax'])
        return image, bb_boxes

    def add_random_shadow(self, image):
        top_y = self.newHeight * np.random.uniform()
        top_x = 0
        bot_x = self.newWidth
        bot_y = self.newWidth * np.random.uniform()
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        shadow_mask = 0 * image_hls[:, :, 1]
        X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

        shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (
            Y_m - top_y) >= 0)] = 1  # random_bright = .25+.7*np.random.uniform()
        if np.random.randint(2) == 1:
            random_bright = .5
            cond1 = shadow_mask == 1
            cond0 = shadow_mask == 0
            if np.random.randint(2) == 1:
                image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
            else:
                image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
        image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)

        return image

    def stretch_image(self, img, bb_boxes_f, scale_range):
        # Stretching augmentation

        bb_boxes_f = bb_boxes_f.copy(deep=True)

        tr_x1 = scale_range * np.random.uniform()
        tr_y1 = scale_range * np.random.uniform()
        p1 = (tr_x1, tr_y1)
        tr_x2 = scale_range * np.random.uniform()
        tr_y2 = scale_range * np.random.uniform()
        p2 = (img.shape[1] - tr_x2, tr_y1)

        p3 = (img.shape[1] - tr_x2, img.shape[0] - tr_y2)
        p4 = (tr_x1, img.shape[0] - tr_y2)

        pts1 = np.float32([[p1[0], p1[1]],
                           [p2[0], p2[1]],
                           [p3[0], p3[1]],
                           [p4[0], p4[1]]])
        pts2 = np.float32([[0, 0],
                           [img.shape[1], 0],
                           [img.shape[1], img.shape[0]],
                           [0, img.shape[0]]]
                          )

        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        img = np.array(img, dtype=np.uint8)

        bb_boxes_f['xmin'] = (bb_boxes_f['xmin'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
        bb_boxes_f['xmax'] = (bb_boxes_f['xmax'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
        bb_boxes_f['ymin'] = (bb_boxes_f['ymin'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]
        bb_boxes_f['ymax'] = (bb_boxes_f['ymax'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]

        return img, bb_boxes_f

