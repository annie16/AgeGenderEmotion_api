import tensorflow as tf
import tensorflow.contrib.slim as slim
import util.OPTS as OPTS
import numpy as np
import os.path
import FaceNet.inception_resnet_v1 as inception_resnet_v1


class FaceNet:
  class OPTS(OPTS.OPTS):
    def __init__(self):
      OPTS.OPTS.__init__(self, 'FaceNet OPTS')
      self.network_name = None
      self.image_mean = np.reshape([117.0, 117.0, 117.0], [1, 1, 1, 3])
      self.weight_path = None
      self.weight_decay = 1e-4

  def __init__(self, opts):
    if opts is None:
      opts = self.OPTS()
    self.opts = opts
    self.opts.assert_all_keys_valid()

  def normalize_input(self, x):
    return x - self.opts.image_mean

  def construct(self):

    self.x = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')


    with tf.device('/CPU:0'):
      self.is_training = tf.Variable(True, dtype=tf.bool, name='is_training')
    self.keep_prob = tf.placeholder(tf.float32)
    self.pre_logits, _ = inception_resnet_v1.inference(self.x - self.opts.image_mean, self.keep_prob,
                                                       phase_train=self.is_training)
    batch_norm_params = {
      # Decay for the moving averages
      'decay': 0.995,
      # epsilon to prevent 0s in variance
      'epsilon': 0.001,
      # force in-place updates of mean and variance estimates
      'updates_collections': None,
      # Moving averages ends up in the trainable variables collection
      'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
      # Only update statistics during training mode
      'is_training': self.is_training
    }

    self.bottleneck = self.pre_logits

    # self.bottleneck = slim.fully_connected(self.pre_logits, 128, activation_fn=None,
    #                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                                   weights_regularizer=slim.l2_regularizer(self.opts.weight_decay),
    #                                   normalizer_fn=slim.batch_norm,
    #                                   normalizer_params=batch_norm_params,
    #                                   scope='Bottleneck', reuse=False)

    #self.bottleneck = slim.fully_connected(self.pre_logits, 1024, activation_fn=None,
    #                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                                       weights_regularizer=slim.l2_regularizer(0.0),
    #                                       normalizer_fn=slim.batch_norm,
    #                                       normalizer_params=batch_norm_params,
    #                                       scope='Bottleneck-1', reuse=False)

  def load_pretrained(self, session, scopes, weight_path, generic=True):
    data_dict = np.load(weight_path).item()
    if generic:
      vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      var_idx = {vars[i].name: i for i in range(len(vars))}
      ops = []
      for key in data_dict.keys():
        for scope in scopes:
          if key.startswith(scope):
            data = data_dict[key]
            scope_key = key
            try:
              var = vars[var_idx[scope_key]]
            except KeyError:
              print ("%s unused (no variable)" % (scope_key))
              continue
            try:
              if len(var.get_shape()) == 5:
                data = np.expand_dims(data, 0)
              elif len(var.get_shape()) == 4 and var.get_shape().as_list()[2] != data.shape[2]:
                data = np.concatenate((data, data), axis=2) / 2.0
              ops.append(var.assign(data))

              print ("%s used" % (scope_key))
            except ValueError:
              print ("%s unused" % (scope_key))
              pass
      session.run(ops)
    else:
      for scope in scopes:
        for op_name in data_dict:
          with tf.variable_scope(scope):
            with tf.variable_scope(op_name, reuse=True):
              if isinstance(data_dict[op_name], list):
                iteritems = {}
                iteritems['weights'] = data_dict[op_name][0]
                iteritems['biases'] = data_dict[op_name][1]
                iteritems = iteritems.iteritems()
              else:
                iteritems = data_dict[op_name].iteritems()
              for param_name, data in iteritems:
                try:
                  var = tf.get_variable(param_name)
                  if len(var.get_shape()) == 5:
                    data = np.expand_dims(data, 0)
                  elif len(var.get_shape()) == 4 and var.get_shape().as_list()[2] != data.shape[2]:
                    data = np.concatenate((data, data), axis=2) / 2.0
                  session.run(var.assign(data))
                  print ("%s/%s used" % (scope, op_name))
                except ValueError:
                  print ("%s/%s unused" % (scope, op_name))
                  pass

  def fc(self, x, dim, name, reuse=False):
    in_shape = x.get_shape()

    s = 1
    for i in range(1, len(in_shape)):
      s *= int(in_shape[i])
    if len(in_shape) >= 4:
      x = tf.reshape(x, [-1, s])
    with tf.variable_scope(name, reuse=reuse):
      weights = tf.get_variable("weights", shape=[s, dim],
                                initializer=tf.contrib.layers.xavier_initializer())
      biases = tf.get_variable("biases", shape=[dim],
                               initializer=tf.constant_initializer(0.0))
      fc = tf.nn.xw_plus_b(x, weights, biases)
      return fc

  def vgg_conv(self, x, dim, num_conv=3, layer_num=None):
    t = x
    for i in range(1, num_conv + 1):
      t = self.vgg_conv2d(t, dim, "conv%d_%d" % (layer_num, i))
    t = self.vgg_pool2d(t, "pool%d" % (layer_num))
    return t

  def vgg_conv2d(self, x, dim, name, reuse=False, trainable=True):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
      weights = tf.get_variable("weights", shape=[3, 3, in_shape[-1], dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=trainable)
      biases = tf.get_variable("biases", shape=[dim],
                               initializer=tf.constant_initializer(0.0),
                               trainable=trainable)
      conv = tf.nn.conv2d(x, weights,
                          strides=[1, 1, 1, 1], padding='SAME')
      return tf.nn.relu(conv + biases)

  def vgg_pool3d(self, in_tensor, name, reuse=False):
    with tf.name_scope(name):
      pool = tf.nn.max_pool3d(in_tensor, ksize=[1, 1, 2, 2, 1],
                              strides=[1, 1, 2, 2, 1], padding='SAME')
      return pool

  def vgg_pool2d(self, in_tensor, name, reuse=False):
    with tf.name_scope(name):
      pool = tf.nn.max_pool(in_tensor, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
      return pool
