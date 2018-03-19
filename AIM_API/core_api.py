import cherrypy as cp
import cv2, av
import numpy as np, math
from StringIO import StringIO
import json
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense,Activation,Convolution2D,MaxPooling2D
from keras import applications
from keras import backend as K
from skimage import io
import tensorflow as tf
from sklearn.preprocessing import normalize
import thread, copy
import types, functools, magic
import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import cv2

import Network



class MethodLogger(object):
    def __init__(self, obj):
        self._obj = obj
        self._log = []
    def __getattr__(self, name):
        value = getattr(self._obj, name)
        if isinstance(value, (types.MethodType, types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType)):
            return functools.partial(self._method, name, value)
        else:
            self._log.append(('__getattr__', (name, ), {}))
            return value
    def _method(self, name, meth, *args, **kwargs):
        self._log.append((name, args, kwargs))
        return meth(*args, **kwargs)
    def _filter(self, type_):
        return [log for log in self._log if log[0] == type_]

class EmotionRecognizer:
    _store  = None
    exposed = True
    CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    BATCH_SIZE = 3
    def __init__(self):
        hidden_dim = 4096
        batch_size = 32
        img_width, img_height = 224, 224
        nb_class = 7
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=False
       # config.gpu_options.per_process_gpu_memory_fraction = 0.7
       # sess = tf.Session(config=config)
        #K.set_session(sess)
       # self.lock = thread.allocate_lock()
        #self.gpu_op_lock = thread.allocate_lock()
        #emotion
        base_model = VGGFace(input_shape=(img_height, img_width, 3), include_top=True, weights=None)
        print ("Emotion Model Successfully Loaded.")
        last_layer = base_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(hidden_dim, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        out = Dense(nb_class, name='fc8')(x)
        out = Activation('relu', name='fc8/softmax')(out)
        model = Model(inputs=base_model.input, outputs=out)
        model.load_weights('weights/fer_full.h5', by_name=True)
        for layer in model.layers[:13]:
            layer.trainable = False
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        self.graph = tf.get_default_graph()
    def GET(self):
        return "face core module"
    def POST(self):
        try:
            vid_data = cp.request.body.read(int(cp.request.headers['Content-Length']))
            sio = StringIO(vid_data)
            f = magic.open( magic.MAGIC_MIME )
            f.load()
            mime_type = f.buffer(vid_data)
            if not mime_type.startswith('video'):
                raise TypeError
            wrapped = MethodLogger(sio)
            vid = []
            container = av.open(wrapped)
            video_stream = next(s for s in container.streams if s.type == 'video')
            try:
                rotation_angle = -int(video_stream.metadata['rotate'])
            except:
                rotation_angle = 0
            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    frame = frame.to_rgb().to_nd_array()
                    if rotation_angle != 0:
                        rows, cols, _ = frame.shape
                        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
                        frame = cv2.warpAffine(frame, M, (cols, rows))
                    vid.append(frame)
        except:
            response = cp.response
            response.status = '406 Not Acceptable'
            response.headers['Content-Type'] = 'text/plain'
            response.headers['Allow'] = 'GET, POST'
            return 'content not acceptable'
        trial_list = []
        crop_batch = []
        face_frame = []
        face_detector = cv2.CascadeClassifier("frontalFace10/haarcascade_frontalface_default.xml")
        for i in range(len(vid)):
            frame = vid[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            measure=[]
            for (x, y, w, h) in faces:
                crop = frame[y: y + h, x: x + w]
                crop = cv2.resize(crop, (224, 224))
                crop = np.expand_dims(crop, axis=0)
                area=w*h
                crop_batch.append(crop)
                trial_list.append((i, (x, y, w, h)))
                measure.append(area)
            if not measure :
                #print ("No faces detected")
                pass
            else:
                minDistance=np.argmax(measure)
                face_frame.append(minDistance)
        """
        w, h = frame.shape[1], frame.shape[0]

        _fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
        _out = cv2.VideoWriter('out/%d.mp4'%np.random.randint(0,1000000), _fourcc, 20.0, (w, h))

        crop_cnt = 0
        for i in range(len(vid)):
            frame = vid[i]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            while crop_cnt < len(trial_list) and trial_list[crop_cnt][0] == i:
                (x, y, w, h) = trial_list[crop_cnt][1]
                cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
                crop_cnt += 1
            _out.write(bgr)
        """
        num_crops = len(face_frame)
        num_batch = int(math.ceil(float(num_crops)/self.BATCH_SIZE))
        if num_crops == 0:
            dict = {'message': 'no face detected'}
            response = cp.response
            response.status = '200 OK'
            response.headers['Content-Type'] = 'application/json'
            response.headers['Allow'] = 'GET, POST'
            return json.dumps(dict)
        with self.graph.as_default():
            start_pnt = 0
            probs, args = [], []
            for b in range(num_batch):
                batch = np.concatenate(crop_batch[start_pnt:start_pnt+self.BATCH_SIZE], axis=0)
                #self.gpu_op_lock.acquire()
                result = self.model.predict(batch)
               # self.gpu_op_lock.release()
                start_pnt += self.BATCH_SIZE
                probs_ = softmax(result)
                args_ = np.argmax(probs_, axis=1)
                probs.append(probs_)
                args.append(args_)
        probs = np.concatenate(probs, axis=0)
        args = np.concatenate(args, axis=0)
        final_result = np.average(probs, axis=0)
        dict = {
                'emotion': {'angry': float(final_result[0]),
                           'disgust': float(final_result[1]),
                           'fear': float(final_result[2]),
                           'happy': float(final_result[3]),
                           'neutral': float(final_result[4]),
                           'sad': float(final_result[5]),
                           'surprise': float(final_result[6])}}
        print dict
        response = cp.response
        response.status = '200 OK'
        response.headers['Content-Type'] = 'application/json'
        response.headers['Allow'] = 'GET, POST'
        return json.dumps(dict)
        #cv2.imwrite('test_.jpg', img)
        #r, buf = cv2.imencode('.jpg', img)
        #b = buf.tobytes()

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
