import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import cv2

import Network

g = tf.Graph()
with g.as_default():
    net_opts = Network.EmbedModel.OPTS()
    net_opts.network_name = "EmbedModel"
    net_opts.apply_dropout = True
    net_opts.loss_type = 'softmax'
    net_opts.distance_metric = 'L2'
    net_opts.net_type = 'FaceNet'
    net_opts.age = False
    net_opts.ged = True
    net_opts.device_string = '/GPU:0'

    net = Network.EmbedModel(net_opts)
    net.construct()

    saver = tf.train.Saver(tf.global_variables())

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
class runner:
    def __init__(self):
        self.sess = tf.Session(config=sess_config, graph=g)

        sess = self.sess
        ckpt_path = 'export/gender/model.ckpt-6747'
        saver.restore(sess, ckpt_path)
        step = int(ckpt_path.split('-')[-1])
        print('[GENDER MODULE] Session restored successfully. step: {0}'.format(step))

    def recognize_age(self, im):

        age_class = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

        im = im[:,:,[2,1,0]]
        im = cv2.resize(im, (160,160))
        im = np.expand_dims(im, 0)

        out = self.sess.run(net.fc8, feed_dict={net.x: im, net.keep_prob: 1., net.is_training: False})

        prob = np.max(softmax(out[0]))


        pred = np.argmax(out, axis=1)
        pred = pred[0]

        #print p

        #pred_svm = self.clf.predict(feat)
        #pred_neigh = self.neigh.predict(feat)
        #print ("predicted %s <- svm, %s <- knn(150)"%
        #       (age_class[pred_svm[0]], age_class[pred_neigh[0]]))
        return pred, prob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()