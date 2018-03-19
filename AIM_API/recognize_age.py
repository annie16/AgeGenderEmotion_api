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
    net_opts.loss_type = 'triplet2'
    net_opts.distance_metric = 'L2'
    net_opts.net_type = 'FaceNet'
    net_opts.age = True
    net_opts.ged = False
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
        ckpt_path = 'export/age/model.ckpt-31303'
        saver.restore(sess, ckpt_path)
        step = int(ckpt_path.split('-')[-1])
        print('[AGE MODULE] Session restored successfully. step: {0}'.format(step))

        with open('export/age/model.ckpt-31303_svm.pkl', 'r') as f:
            self.clf = pickle.load(f)

        with open('export/age/model.ckpt-31303_neigh.pkl', 'r') as f:
            self.neigh = pickle.load(f)

        with open('export/age/model.ckpt-31303_neigh2.pkl', 'r') as f:
            self.neigh2 = pickle.load(f)

        with open('export/age/model.ckpt-31303_neigh3.pkl', 'r') as f:
            self.neigh3 = pickle.load(f)

        with open('export/age/model.ckpt-31303_a_labels.pkl', 'r') as f:
            a_labels = pickle.load(f)
            self.a_labels = np.array(a_labels)

    def recognize_age(self, im):

        age_class = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

        im = im[:,:,[2,1,0]]
        im = cv2.resize(im, (160,160))
        im = np.expand_dims(im, 0)
        im_flip = im[:,:,::-1]
        feat = self.sess.run(net.feat_norm, feed_dict={net.x: im, net.keep_prob: 1., net.is_training: False})

        ind = self.neigh2.kneighbors(feat, n_neighbors=64, return_distance=False)
        sel = self.a_labels[ind]
        sel[sel<20] = 20
        p = np.average(sel)
        
        feat = self.sess.run(net.feat_norm, feed_dict={net.x: im_flip, net.keep_prob: 1., net.is_training: False})

        ind = self.neigh2.kneighbors(feat, n_neighbors=64, return_distance=False)
        sel = self.a_labels[ind]
        sel[sel<20] = 20
        p2 = np.average(sel)
        #p2 = np.average(self.a_labels[ind])
        
        p = (p+p2)/2
        #print p

        #pred_svm = self.clf.predict(feat)
        #pred_neigh = self.neigh.predict(feat)
        #print ("predicted %s <- svm, %s <- knn(150)"%
        #       (age_class[pred_svm[0]], age_class[pred_neigh[0]]))
        return p
