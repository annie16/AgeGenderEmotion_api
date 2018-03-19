import tensorflow as tf
import numpy as np
from pretrained.face_net import FaceNet
import util.OPTS as OPTS

class modified_fnet(FaceNet):
  class OPTS(OPTS.OPTS):
    LOSS_TYPE = ['triplet1', 'triplet2', 'softmax']
    INPUT_SIZE = 160
    def __init__(self):
      OPTS.OPTS.__init__(self, 'FaceNet OPTS')
      self.network_name = None
      self.image_mean = np.reshape([117.0, 117.0, 117.0], [1, 1, 1, 3])
      self.weight_path = None
      self.apply_dropout = None
      self.loss_type = 'triplet1'
      self.num_classes = None

  def __init__(self, opts):
    FaceNet.__init__(self, opts)

  def construct(self):
    FaceNet.construct(self)

    if self.opts.loss_type == self.OPTS.LOSS_TYPE[0] or self.opts.loss_type == self.OPTS.LOSS_TYPE[1]:
      self.l2_norm = tf.nn.l2_normalize(self.bottleneck, 1, name='normalize_l2')
      dist_mat = tf.matmul(self.l2_norm, self.l2_norm, transpose_a=False, transpose_b=True)

      gt_idx = tf.argmax(self.y_age, axis=1)
      gt_idx = tf.expand_dims(gt_idx, axis=1)
      gt_idx = tf.cast(gt_idx, tf.float32)
      gt_1 = tf.tile(gt_idx, [1, tf.shape(gt_idx)[0]])
      gt_2 = tf.transpose(gt_1, [1,0])

      dist_gt = tf.abs(gt_1 - gt_2)

      pos_dist = tf.equal(dist_gt, 0.)
      neg_dist = tf.logical_not(pos_dist)

      def cart_mat(a, remove_diag=True, pos_dist=None, neg_dist=None):
        s = tf.shape(a)

        if remove_diag:
          diag = tf.tile(tf.constant(1e+6, shape=[1], dtype=tf.float32), [s[1]])
          a_1 = tf.matrix_set_diag(a, diag)
          a_2 = tf.matrix_set_diag(a, -diag)
        else:
          a_1, a_2 = a, a

        if neg_dist != None:
          a_1 = a_1 + tf.cast(neg_dist, tf.float32) * 1e+6
        if pos_dist != None:
          a_2 = a_2 - tf.cast(pos_dist, tf.float32) * 1e+6


        a_1 = tf.expand_dims(a_1, axis=1)
        a_2 = tf.expand_dims(a_2, axis=2)

        a_1 = tf.tile(a_1, [1,s[1],1])
        a_2 = tf.tile(a_2, [1, 1, s[1]])

        a_1 = tf.reshape(a_1, [s[0], -1])
        a_2 = tf.reshape(a_2, [s[0], -1])

        return a_2 - a_1

      if self.opts.loss_type == self.OPTS.LOSS_TYPE[0]:
        tri_loss = cart_mat(dist_mat, remove_diag=True)
      else:
        tri_loss = cart_mat(dist_mat, remove_diag=True, pos_dist=pos_dist, neg_dist=neg_dist)

      gt_dist_coef = tf.maximum(0., cart_mat(dist_gt, remove_diag=False))

      final_triplet_loss_mat = tf.multiply(tf.maximum(0., tri_loss + 0.1), gt_dist_coef)
      self.loss = tf.reduce_mean(final_triplet_loss_mat)

      with tf.variable_scope('fc8-1'):
        weights = tf.get_variable("weights", shape=[128, self.opts.num_classes],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", shape=[self.opts.num_classes],
                                 initializer=tf.constant_initializer(0.0))
        fc = tf.nn.xw_plus_b(self.bottleneck, weights, biases)

      self.fc8 = fc
      self.softmax = tf.nn.softmax(self.fc8, name='softmax')

      self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=self.y_age))
      self.correct_pred = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(self.y_age, 1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

      self.softmax_trainer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(self.class_loss, var_list=[weights, biases])

    elif self.opts.loss_type == self.OPTS.LOSS_TYPE[2]:
      self.fc8 = self.fc(self.bottleneck, self.opts.num_classes, 'fc8-1')
      self.softmax = tf.nn.softmax(self.fc8, name='softmax')

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=self.y_ged))
      self.correct_pred = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(self.y_ged, 1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


class modified_fnet_feat(FaceNet):
  class OPTS(OPTS.OPTS):
    def __init__(self):
      OPTS.OPTS.__init__(self, 'FaceNet OPTS')
      self.network_name = None
      self.image_mean = np.reshape([117.0, 117.0, 117.0], [1, 1, 1, 3])


  def __init__(self, opts):
    FaceNet.__init__(self, opts)

  def construct(self):
    FaceNet.construct(self)
    self.y_age = tf.placeholder(tf.float32, shape=[None, None])
    self.y_ged = tf.placeholder(tf.float32, shape=[None, 2])