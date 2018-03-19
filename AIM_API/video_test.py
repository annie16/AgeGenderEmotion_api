import cherrypy as cp
import cv2
import numpy as np
from StringIO import StringIO
import json
import av

class FaceCore:
    _store  = None
    exposed = True

    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier("frontalFace10/haarcascade_frontalface_default.xml")

    def GET(self):
        return "face core module"

    def POST(self):
        img_data = cp.request.body.read(int(cp.request.headers['Content-Length']))
        sio = StringIO(img_data)

        container = av.open(sio)
        video_stream = next(s for s in container.streams if s.type == 'video')
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                frame.to_image().save('frame-%02d.jpg' % frame.index)

        return 'success'
        img = cv2.imdecode(jpg_bytes, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #cv2.imwrite('test_.jpg', img)
        r, buf = cv2.imencode('.jpg', img)
        b = buf.tobytes()

        response = cp.response
        response.status = '200 OK'
        response.headers['Content-Type'] = 'application/octet-stream'

        dict = {'a':100, 'type':type}
        return b

