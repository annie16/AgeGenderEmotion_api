import random
import string
import cv2
import numpy as np

import cherrypy
import urllib

API_ACCESS_KEY = 'abcde'

cherrypy.config.update({'server.socket_host': "143.248.136.158", 'server.socket_port': 6015})

@cherrypy.expose
class ImageRepeater(object):
    @cherrypy.tools.accept(media='text/html')
    def GET(self):
        str = \
        """
        <html>
        <head>
        <title>TEST</title>
        </head>
        <body>
            <form action="/" method="POST">
            key <br>
            <input type="text" name="key">
            <br>
            file
            <br>
            <input type="file">
            <input type="submit">
            </form>
        </body>
        </html>
        """
        return str

    def POST(self, key):
        if key == API_ACCESS_KEY:
            img_data = cherrypy.request.body.read(int(cherrypy.request.headers['Content-Length']))
            img_byte = bytearray()
            img_byte.extend(img_data)

            jpg_bytes = np.asarray(img_byte, dtype=np.uint8)
            img = cv2.imdecode(jpg_bytes, cv2.IMREAD_UNCHANGED)
            cv2.imwrite('test_.jpg', img)
            return 'success'
        else:
            return 'invalid access: wrong api key.'


if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')],
        }
    }
    cherrypy.quickstart(ImageRepeater(), '/', conf)