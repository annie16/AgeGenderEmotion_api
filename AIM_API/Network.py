import tensorflow as tf
import numpy as np
from network.face_net import modified_fnet_feat

from util.OPTS import OPTS


class EmbedModel:
    class OPTS(OPTS):
        LOSS_TYPE = ['triplet1', 'triplet2', 'softmax']
        INPUT_SIZE = 160

        def __init__(self):
            OPTS.__init__(self, 'EmbedModel OPTS')
            self.network_name = None
            self.apply_dropout = None
            self.loss_type = None
            self.distance_metric = None
            self.net_type = None
            self.age = None
            self.ged = None
            self.device_string = None
            self.age_is_val = False
            self.coef_op = 'linear'
            self.weight_decay = 1e-4

    def __init__(self, opts=None):
        if opts is None:
            opts = self.OPTS()
        opts.assert_all_keys_valid()
        self.opts = opts

    def construct(self):
        self.feat = self.__embed_net()
        self.feat_norm = tf.nn.l2_normalize(self.feat, 1)
        if self.opts.age:
            aff = self.__make_gt_dist_mat(self.y_age)
        elif self.opts.ged:
            aff = self.__make_gt_dist_mat(self.y_ged)
        else:
            raise

        if self.opts.loss_type.startswith("triplet"):
            pass
        else:
            if self.opts.age:
                gt = self.y_age
                num_class = 8
            elif self.opts.ged:
                gt = self.y_ged
                num_class = 2
            else:
                raise

            self.fc8 = _fc(self.feat, num_class, 'fc8-1')
            self.softmax = tf.nn.softmax(self.fc8, name='softmax')

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=gt))
            self.correct_pred = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(gt, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def __embed_net(self):
        """
        :param input: tf.Tensor  # input tensor
        :return: tf.Tensor  # embed feature
        """
        if self.opts.net_type == "VGGFace":
            net_opts = modified_vgg_feat.OPTS()
            net_opts.age = self.opts.age
            net_opts.network_name = 'vgg_face'
            net_opts.weight_path = 'pretrained/vgg-face.mat'
            net_opts.apply_dropout = self.opts.apply_dropout
            feat_net = modified_vgg_feat(net_opts)
            feat_net.construct()
            feat = feat_net.fc7

        elif self.opts.net_type == "FaceNet":
            net_opts = modified_fnet_feat.OPTS()
            net_opts.network_name = 'FaceNet'
            net_opts.weight_path = 'pretrained/FaceNet/20170216-091149/model-20170216-091149.ckpt-250000.npy'
            feat_net = modified_fnet_feat(net_opts)
            feat_net.construct()
            feat = feat_net.bottleneck
        else:
            raise

        self.feat_net = feat_net
        self.x = feat_net.x
        self.keep_prob = feat_net.keep_prob
        self.y_age = feat_net.y_age
        self.y_ged = feat_net.y_ged
        self.is_training = feat_net.is_training

        return feat

    def __make_gt_dist_mat(self, ground):
        with tf.name_scope("make_gt_dist_mat"):
            if self.opts.age_is_val:
                gt_idx = ground
            else:
                gt_idx = tf.argmax(ground, axis=1)
                gt_idx = tf.expand_dims(gt_idx, axis=1)
                gt_idx = tf.cast(gt_idx, tf.float32)
            gt_1 = tf.tile(gt_idx, [1, tf.shape(gt_idx)[0]])
            gt_2 = tf.transpose(gt_1, [1, 0])

            dist_gt = tf.abs(gt_1 - gt_2)
        return dist_gt

    def load_pretrained(self, sess):
        if self.opts.net_type == "VGGFace":
            self.feat_net.load_pretrained(sess)
        elif self.opts.net_type == "FaceNet":
            self.feat_net.load_pretrained(sess, [''], weight_path=self.feat_net.opts.weight_path, generic=True)
        else:
            raise
def _fc(bottom, output_dim, name, reuse=False, bias=True):
    with tf.variable_scope(name, reuse=reuse):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = tf.get_variable(name='weights', shape=[dim, output_dim],
                                  initializer=tf.contrib.layers.xavier_initializer())

        if bias:
            b = tf.get_variable(name='biases', shape=[output_dim],
                                   initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

            fc = tf.nn.xw_plus_b(x, weights, b, name="fc")
        else:
            fc = tf.matmul(x, weights, name="fc")
        return fc
