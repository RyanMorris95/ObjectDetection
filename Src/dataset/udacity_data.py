import numpy as np
import sklearn
import pandas as pd
from src.dataset.data_augmentation import DataAugmentation
import cv2
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET

class Voc_Dataset:
    def __init__(self, mc, sess):
        self.data = None
        self.mc = mc
        self.data_augmentation = DataAugmentation(1920, 1080, self.mc.IMAGE_WIDTH, self.mc.IMAGE_HEIGHT)
        self.dir_label = ['D:\MechatronicsDataset']
        self._getData()
        self.sess = sess

    def _getData(self):
        """"
        # Dataset 1
        df_files1 = pd.read_csv(self.dir_label[0] + '\labels.csv', header=0)
        df_vehicles1 = df_files1[(df_files1['Label'] == 'Car') | (df_files1['Label'] == 'Truck')].reset_index()
        df_vehicles1 = df_vehicles1.drop('index', 1)
        df_vehicles1['File_Path'] = self.dir_label[0] + '/' + df_vehicles1['Frame']
        df_vehicles1 = df_vehicles1.drop('Preview URL', 1)

        # Dataset 2
        df_files2 = pd.read_csv('C:\\squeezeDet-master\\data\\object-dataset\\labels.csv', sep=' ',
                                names=["Frame", "xmin", "xmax", "ymin", "ymax", "ind", "Label"])
        df_vehicles2 = df_files2[(df_files2['Label'] == 'car') | (df_files2['Label'] == 'truck')].reset_index()
        df_vehicles2 = df_vehicles2.drop('index', 1)
        df_vehicles2 = df_vehicles2.drop('ind', 1)
        df_vehicles2['File_Path'] = self.dir_label[1] + '/' + df_vehicles2['Frame']

        # Create combined dataset
        self.data = pd.concat([df_vehicles1, df_vehicles2]).reset_index()
        self.data = self.data.drop('index', 1)
        self.data.columns = ['File_Path', 'Frame', 'Label', 'ymin', 'xmin', 'ymax', 'xmax']
        self.data.head()
        """
        xml_files = glob.glob('D:\MechatronicsDataset\*.xml')
        self.data = pd.DataFrame(columns=['File_Path', 'Frame', 'Label', 'ymin', 'xmin', 'ymax', 'xmax'])
        annotation_dicts = []
        # parse xml
        for file in xml_files:
            tree = ET.parse(file)
            root = tree.getroot()

            file_path = root.find('path').text
            frame = root.find('filename').text
            for object in root.iter('object'):
                bndbox = object.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                ymax = int(bndbox.find('ymax').text)
                xmax = int(bndbox.find('xmax').text)
                label = object.find('name').text
                annot_dict = {'File_Path': file_path, 'Frame': frame, 'Label': label, 'ymin': ymin, 'xmin': xmin,
                               'ymax': ymax, 'xmax': xmax}
                annotation_dicts.append(annot_dict)

        self.data = pd.DataFrame(annotation_dicts)


    def _getBoxesList(self, bb_boxes):
        boxes = []
        labels = []
        for i in range(len(bb_boxes)):
            bb_box_i = [bb_boxes.iloc[i]['xmin'], bb_boxes.iloc[i]['ymin'],
                        bb_boxes.iloc[i]['xmax'], bb_boxes.iloc[i]['ymax']]

            if bb_boxes.iloc[i]['xmin'] > bb_boxes.iloc[i]['xmax']:
                cx = ((bb_boxes.iloc[i]['xmin'] - bb_boxes.iloc[i]['xmax']) / 2.) + bb_boxes.iloc[i]['xmax']
                cy = ((bb_boxes.iloc[i]['ymin'] - bb_boxes.iloc[i]['ymax']) / 2.) + bb_boxes.iloc[i]['ymax']
                width = (bb_boxes.iloc[i]['xmin'] - bb_boxes.iloc[i]['xmax'])
                height = (bb_boxes.iloc[i]['ymin'] - bb_boxes.iloc[i]['ymax'])
            else:
                cx = ((bb_boxes.iloc[i]['xmax'] - bb_boxes.iloc[i]['xmin']) / 2.) + bb_boxes.iloc[i]['xmin']
                cy = ((bb_boxes.iloc[i]['ymax'] - bb_boxes.iloc[i]['ymin']) / 2.) + bb_boxes.iloc[i]['ymin']
                width = (bb_boxes.iloc[i]['xmax'] - bb_boxes.iloc[i]['xmin'])
                height = (bb_boxes.iloc[i]['ymax'] - bb_boxes.iloc[i]['ymin'])

            if cx < 0:
                cx = abs(cx)
            if cy < 0:
                cy = abs(cy)
            if width < 0:
                width = abs(width)
            if height < 0:
                height = abs(height)

            label = bb_boxes.iloc[i]['Label']

            boxes.append([cx, cy, width, height])
            labels.append(label)

        return bb_box_i, boxes, labels

    def _get_image_name(self, df, ind, augmentation=False):
        # Get image by name
        file_name = self.data['File_Path'][ind]

        img = cv2.imread(file_name)
        img_size = np.shape(img)
        img -= np.mean(img, axis=(0, 1, 2), dtype=np.uint8)
        cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = cv2.resize(img, (self.mc.IMAGE_WIDTH, self.mc.IMAGE_HEIGHT))

        name_str = self.data['Frame'][ind]
        #name_str = name_str[0]
        # print(name_str)
        # print(file_name)

        #df.groupby(df.columns, axis=1).sum()
        bb_boxes = df[df['Frame'] == name_str].reset_index()

        img_size_post = np.shape(img)
        x_scale = img_size_post[1] / img_size[1]
        y_scale = img_size_post[0] / img_size[0]
        bb_boxes['xmin'] = np.round(bb_boxes['xmin'] * x_scale)
        bb_boxes['xmax'] = np.round(bb_boxes['xmax'] * x_scale)
        bb_boxes['ymin'] = np.round(bb_boxes['ymin'] * y_scale)
        bb_boxes['ymax'] = np.round(bb_boxes['ymax'] * y_scale)
        bb_boxes['Area'] = (bb_boxes['xmax'] - bb_boxes['xmin']) * (bb_boxes['ymax'] - bb_boxes['ymin'])

        trans_range = 50
        scale_range = 50

        if augmentation:
            #img = cv2.resize(img, (self.mc.IMAGE_WIDTH, self.mc.IMAGE_HEIGHT))
            img, bb_boxes = self.data_augmentation.trans_image(img, bb_boxes, trans_range)
            img, bb_boxes = self.data_augmentation.stretch_image(img, bb_boxes, scale_range)
            img, bb_boxes = self.data_augmentation.random_flip(img, bb_boxes)
            img = self.data_augmentation.add_random_shadow(img)

        for i in range(0, len(bb_boxes)):
            xmin, xmax, ymin, ymax = bb_boxes.iloc[i]['xmin'], bb_boxes.iloc[i]['xmax'], bb_boxes.iloc[i]['ymin'], \
                                     bb_boxes.iloc[i]['ymax']
            if xmin <= 0:
                bb_boxes.set_value(i, 'xmin', 1)
            if xmax >= self.mc.IMAGE_WIDTH:
                bb_boxes.set_value(i, 'xmax', self.mc.IMAGE_WIDTH-1)
            if ymin <= 0:
                bb_boxes.set_value(i, 'ymin', 1)
            if ymax >= self.mc.IMAGE_HEIGHT:
                bb_boxes.set_value(i, 'ymax', self.mc.IMAGE_HEIGHT-1)
            if xmin >= self.mc.IMAGE_WIDTH:
                bb_boxes.set_value(i, 'xmin', self.mc.IMAGE_WIDTH-1)
            if xmax <= 0:
                bb_boxes.set_value(i, 'xmax', 1)
            if ymin >= self.mc.IMAGE_HEIGHT:
                bb_boxes.set_value(i, 'ymin', self.mc.IMAGE_HEIGHT-1)
            if ymax <= 0:
                bb_boxes.set_value(i, 'ymax', 1)

        return name_str, img, bb_boxes

    def generate_train_batch(self):
        """
        Returns:
            image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
            label_per_batch: labels. Shape: batch_size x object_num
            delta_per_batch: bounding box deltas. Shape: batch_size x object_num x [dx, dy, dw, dh]
            aidx_per_batch: index of anchors that are responsible for prediction.
                            Shape: batch_size x object_num
            bbox_per_batch: scaled bounding boxes. Shape: batch_size x object_num x [cx, cy, w, h]
        """

        image_per_batch, label_per_batch, bbox_per_batch, delta_per_batch, aidx_per_batch = [], [], [], [], []

        batch_images = np.zeros((self.mc.BATCH_SIZE, self.mc.IMAGE_WIDTH, self.mc.IMAGE_HEIGHT, 3))

        avg_ious = 0.
        num_objects = 0.
        max_iou = 0.0
        min_iou = 1.0
        num_zero_iou_obj = 0

        while 1:
            for i_batch in range(self.mc.BATCH_SIZE):
                i_line = np.random.randint(len(self.data))
                name_str, img, bb_boxes = self._get_image_name(self.data, i_line,
                                                               augmentation=True)
                # Create list of boxes
                bb_box_i, gt_bbox, labels = self._getBoxesList(bb_boxes)
                orig_h, orig_w, _ = np.shape(img)

                bbox_per_batch.append(gt_bbox)
                label_per_batch.append(labels)
                image_per_batch.append(img)

                aidx_per_image, delta_per_image = [], []
                aidx_set = set()
                for i in range(len(gt_bbox)):
                    overlaps = self.data_augmentation.batch_iou(self.mc.ANCHOR_BOX, gt_bbox[i])

                    aidx = len(self.mc.ANCHOR_BOX)
                    for ov_idx in np.argsort(overlaps)[::-1]:
                        if overlaps[ov_idx] <= 0:
                            if self.mc.DEBUG_MODE:
                                min_iou = min(overlaps[ov_idx], min_iou)
                                num_objects += 1
                                num_zero_iou_obj += 1
                            break
                        if ov_idx not in aidx_set:
                            aidx_set.add(ov_idx)
                            aidx = ov_idx
                            if self.mc.DEBUG_MODE:
                                max_iou = max(overlaps[ov_idx], max_iou)
                                min_iou = min(overlaps[ov_idx], min_iou)
                                avg_ious += overlaps[ov_idx]
                                num_objects += 1
                            break

                    if aidx == len(self.mc.ANCHOR_BOX):
                        # even the largeset available overlap is 0, thus, choose one with the
                        # smallest square distance
                        dist = np.sum(np.square(gt_bbox[i] - self.mc.ANCHOR_BOX), axis=1)
                        for dist_idx in np.argsort(dist):
                            if dist_idx not in aidx_set:
                                aidx_set.add(dist_idx)
                                aidx = dist_idx
                                break

                    box_cx, box_cy, box_w, box_h = gt_bbox[i]

                    delta = [0] * 4
                    ##################
                    # All this was good
                    # print ("box_w: {}, box_h: {}, anchor[aidx][2]: {}, anchor[aidx][3]: {}".format(box_w, box_h, mc.ANCHOR_BOX[aidx][2], mc.ANCHOR_BOX[aidx][3]))
                    ##################
                    # if box_w == 0:
                    #     box_w = 1
                    # if box_h == 0:
                    #     box_h = 1
                    delta[0] = np.clip(((box_cx - self.mc.ANCHOR_BOX[aidx][0]) / box_w), 1e-10, 1.0)
                    delta[1] = np.clip(((box_cy - self.mc.ANCHOR_BOX[aidx][1]) / box_h), 1e-10, 1.0)
                    # # MAyber try: (tf.clip_by_value(y_conv,1e-10,1.0)))
                    # delta[2] = np.where(mc.ANCHOR_BOX[aidx][2] > 0, (np.log(tf.clip_by_value((box_w / mc.ANCHOR_BOX[aidx][2]), 1e-10, 1.0), 1)
                    delta[2] = np.log(np.clip((box_w / self.mc.ANCHOR_BOX[aidx][2]), 1e-10, 1.0))
                    # #delta[3] = np.where(mc.ANCHOR_BOX[aidx][3] > 0, (np.log(box_h / mc.ANCHOR_BOX[aidx][3])), 1)
                    delta[3] = np.log(np.clip((box_h / self.mc.ANCHOR_BOX[aidx][3]), 1e-10, 1.0))
                    # #print("delta[0]: {}, delta[1]: {}, delta[2]: {}, delta[3]: {}".format(delta[0], delta[1],
                    #                                                                               #delta[2],
                    #                                                                               #delta[3]))
                    # box_cx, box_cy, box_w, box_h = gt_bbox[i]
                    # delta = [0] * 4
                    # delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0]) / box_w
                    ##delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1]) / box_h
                    # delta[2] = np.log(box_w / mc.ANCHOR_BOX[aidx][2])
                    # delta[3] = np.log(box_h / mc.ANCHOR_BOX[aidx][3])

                    aidx_per_image.append(aidx)
                    delta_per_image.append(delta)

                delta_per_batch.append(delta_per_image)
                aidx_per_batch.append(aidx_per_image)

            return sklearn.utils.shuffle(image_per_batch, label_per_batch, delta_per_batch, aidx_per_batch,
                                         bbox_per_batch)


def plot_bbox(bb_boxes, ind_bb, color='r', linewidth=1):
    ### Plot bounding box

    bb_box_i = [bb_boxes.iloc[ind_bb]['xmin'],
                bb_boxes.iloc[ind_bb]['ymin'],
                bb_boxes.iloc[ind_bb]['xmax'],
                bb_boxes.iloc[ind_bb]['ymax']]
    plt.plot([bb_box_i[0], bb_box_i[2], bb_box_i[2],
              bb_box_i[0], bb_box_i[0]],
             [bb_box_i[1], bb_box_i[1], bb_box_i[3],
              bb_box_i[3], bb_box_i[1]],
             color, linewidth=linewidth)


def plot_box(bb_box_i, ind_bb, coords, color='r', linewidth=1):
    ### Plot bounding box

    bb_box_i = [coords[0],
                coords[2],
                coords[1],
                coords[3]]
    plt.plot([bb_box_i[0], bb_box_i[2], bb_box_i[2],
              bb_box_i[0], bb_box_i[0]],
             [bb_box_i[1], bb_box_i[1], bb_box_i[3],
              bb_box_i[3], bb_box_i[1]],
             color, linewidth=linewidth)


def plot_im_bbox(im, boxes):
    ### Plot image and bounding box
    plt.imshow(im)
    for i in range(len(boxes)):
        xmin = boxes[i][0] - (boxes[i][2] / 2.)
        xmax = boxes[i][0] + (boxes[i][2] / 2.)
        ymin = boxes[i][1] - (boxes[i][3] / 2.)
        ymax = boxes[i][1] + (boxes[i][3] / 2.)
        plot_box(boxes, i, [xmin, xmax, ymin, ymax], 'r')

        bb_box_i = [xmin, ymin, xmax, ymax]
        plt.plot(bb_box_i[0], bb_box_i[1], 'rs')
        plt.plot(bb_box_i[2], bb_box_i[3], 'bs')
    plt.axis('off');


if __name__ == '__main__':
    with tf.Session() as sess:
        tf.global_variables_initializer()
        mecha_config = mecha_squeezeDet_config.mecha_squeezeDet_config()

        voc_dataset = Voc_Dataset(mecha_config, sess)
        image_per_batch, label_per_batch, delta_per_batch, aidx_per_batch, bbox_per_batch = voc_dataset.generate_train_batch()

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.image as mimage

        gs = gridspec.GridSpec(5, 8, top=1., bottom=0., right=1., left=0., hspace=0, wspace=0)
        print (bbox_per_batch)
        count = 0
        for g in gs:
            ax = plt.subplot(g)
            ax.imshow(image_per_batch[count])
            plot_im_bbox(image_per_batch[count], bbox_per_batch[count])
            ax.set_xticks([])
            ax.set_yticks([])
            count += 1


        #plt.imshow(image_per_batch[0])
        plt.show()
