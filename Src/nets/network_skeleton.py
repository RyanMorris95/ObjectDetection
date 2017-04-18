import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import pickle
import cv2

class NetworkSkeleton:
    def __init__(self, config):
        self.yolo2_loss = None
        self.yolo_loss = None
        self.config = config
        self.fetch = None
        self.saver = None
        self.sess = None
        self.input = None
        self.out = None

    def _expit_tensor(self, x):
        return 1. / (1. + tf.exp(-x))

    def yolo_loss(self, net_out):
        """
        Takes net.out and calculates the loss.
        Credit given to: https://github.com/thtrieu/darkflow/blob/master/net/yolo/train.py
        :param net_out: Output of the yolo network
        :return: loss
        """
        # meta
        sprob = float(self.config.class_scale)
        sconf = float(self.config.object_scale)
        snoob = float(self.config.noobject_scale)
        scoor = float(self.config.coord_scale)
        S, B, C = self.config.side, self.config.num, self.config.classes
        SS = S * S # number of grid cells

        print('{} loss hyper-parameters:'.format(m['model']))
        print('\tside    = {}'.format(m['side']))
        print('\tbox     = {}'.format(m['num']))
        print('\tclasses = {}'.format(m['classes']))
        print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

        size1 = [None, SS, C]
        size2 = [None, SS, B]

        # return the below placeholders
        _probs = tf.placeholder(tf.float32, size1)
        _confs = tf.placeholder(tf.float32, size2)
        _coord = tf.placeholder(tf.float32, size2 + [4])
        # weights term for L2 loss
        _proid = tf.placeholder(tf.float32, size1)
        # material calculating IOU
        _areas = tf.placeholder(tf.float32, size2)
        _upleft = tf.placeholder(tf.float32, size2 + [2])
        _botright = tf.placeholder(tf.float32, size2 + [2])

        self.placeholders = {
            'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
            'areas':_areas, 'upleft':_upleft, 'botright':_botright
        }

        # Extract the coordinate prediction from net.out
        coords = net_out[:, SS * (C + B):]
        coords = tf.reshape(coords, [-1, SS, B, 4])
        wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell
        area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2
        centers = coords[:,:,:,0:2] # [batch, SS, B, 2]
        floor = centers - (wh * .5) # [batch, SS, B, 2]
        ceil  = centers + (wh * .5) # [batch, SS, B, 2]

        # calculate the intersection areas
        intersect_upleft   = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil , _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)

        # take care of the weight terms
        conid = snoob * (1. - confs) + sconf * confs
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
        cooid = scoor * weight_coo
        proid = sprob * _proid

        # flatten 'em all
        probs = slim.flatten(_probs)
        proid = slim.flatten(proid)
        confs = slim.flatten(confs)
        conid = slim.flatten(conid)
        coord = slim.flatten(_coord)
        cooid = slim.flatten(cooid)

        self.fetch += [probs, confs, conid, cooid, proid]
        true = tf.concat([probs, confs, coord], 1)
        wght = tf.concat([proid, conid, cooid], 1)
        print('Building {} loss'.format(m['model']))
        loss = tf.pow(net_out - true, 2)
        loss = tf.multiply(loss, wght)
        loss = tf.reduce_sum(loss, 1)
        self.loss = .5 * tf.reduce_mean(loss)
        tf.summary.scalar('{} loss'.format(m['model']), self.loss)

    def yolo2_loss(self, net_out):
        """
        Takes net.out and placeholders value returned in batch to
        build train_op and loss
        Credits: https://github.com/thtrieu/darkflow/blob/master/net/yolov2/train.py
        :param net_out: output of the yolo network
        :return: loss
        """
        sprob = float(self.config.scale)
        sconf = float(self.config.object_scale)
        snoob = float(self.config.noobject_scale)
        scoor = float(self.config.coord_scale)
        H, W, _ = self.config.out_size
        B, C = self.config.num, self.config.classes
        HW = H * W
        anchors = self.config.anchors

        print('{} loss hyper-parameters:'.format(self.config.name))
        print('\tH       = {}'.format(H))
        print('\tW       = {}'.format(W))
        print('\tbox     = {}'.format(self.config.num))
        print('\tclasses = {}'.format(self.config.classes))
        print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

        size1 = [None, HW, B, C]
        size2 = [None, HW, B]

        # return the below placeholders
        _probs = tf.placeholder(tf.float32, size1)
        _confs = tf.placeholder(tf.float32, size2)
        _coord = tf.placeholder(tf.float32, size2 + [4])
        # weights term for L2 loss
        _proid = tf.placeholder(tf.float32, size1)
        # material calculating IOU
        _areas = tf.placeholder(tf.float32, size2)
        _upleft = tf.placeholder(tf.float32, size2 + [2])
        _botright = tf.placeholder(tf.loat32, size2 + [2])

        self.placeholders = {
            'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid'_proid,
            'areas': _areas, 'upleft': _upleft, 'botright': _botright
        }

        # Extract the coordinate prediction from net.out
        net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)])
        coords = net_out_reshape[:, :, :, :, :4]
        coords = tf.reshape(coords, [-1, H * W, B, 4])
        adjusted_coords_xy = self.expit_tensor(coords[:, :, :, 0:2])
        adjusted_coords_wh = tf.sqrt(
            tf.exp(coords[:, :, :, 2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_c = self.expit_tensor(net_out_reshape[:, :, :, :, 4])
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

        adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
        adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

        adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

        wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]
        floor = centers - (wh * .5)
        ceil = centers + (wh * .5)

        # calculate the intersection areas
        intersect_upleft = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil, _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)

        # take care of the weight terms
        conid = snoob * (1. - confs) + sconf * confs
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
        cooid = scoor * weight_coo
        weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
        proid = sprob * weight_pro

        self.fetch += [_probs, confs, conid, cooid, proid]
        true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs], 3)
        wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid], 3)

        print('Building {} loss'.format(self.config.name))
        loss = tf.pow(adjusted_net_out - true, 2)
        loss = tf.multiply(loss, wght)
        loss = tf.reshape(loss, [-1, H * W * B * (4 + 1 + C)])
        loss = tf.reduce_sum(loss, 1)
        self.yolo_loss = .5 * tf.reduce_mean(loss)
        tf.summary.scalar('{} loss'.format(self.config.name), self.yolo_loss)
        return self.yolo_loss

    def save_ckpt(self, step, loss_profile):
        """
        Saves the model checkpoint.
        :param step: Step number of training.
        :param loss_profile: The current loss
        :return:
        """
        file = '{}-{}{}'
        model = self.config.name

        profile = file.format(model, step, '.profile')
        profile = os.path.join(self.config.checkpoint_path, profile)
        with open(profile, 'wb') as profile_ckpt:
            pickle.dump(loss_profile, profile_ckpt)

        ckpt = file.format(model, step, '')
        ckpt = os.path.join(self.config.checkpoint_path, profile)

        print('Checkpoint at step {}'.format(step))
        self.saver.save(self.sess, ckpt)

    def return_predict(self, im):
        """
        Returns predicted boxes from an image.
        :param im: numpy image
        :return: output boxes
        """
        assert isinstance(im, np.ndarray), 'Image is not a np.ndarray'

        h, w, _ = im.shape
        im = cv2.resize(im, self.config.input_size)
        this_input = np.expand_dims(im, 0)
        feed_dict = {self.input : this_input}

        out = self.sess.run(self.out, feed_dict)[0]
        boxes =



