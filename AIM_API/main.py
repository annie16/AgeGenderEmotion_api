#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cherrypy
import random
import string
import core_api
import db

# Use http://websistent.com/tools/htdigest-generator-tool/ to generate proper hashes
userpassdict  = db.get_users()
get_ha1       = cherrypy.lib.auth_digest.get_ha1_dict(userpassdict)

config = {
  'global' : {
        'server.socket_host': '143.248.136.158',
        'server.socket_port': 8080,
        'server.max_request_body_size' : 0,
        'server.socket_timeout' : 60,
        'server.ssl_module':'pyopenssl',
        'server.ssl_certificate':"/home/wbim/.ssl/server.crt",
        'server.ssl_private_key':"/home/wbim/.ssl/server.key",
  },
  '/' : {
    # HTTP verb dispatcher
    'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
    # JSON response
    # 'tools.json_out.on' : True,
    # Digest Auth
    'tools.auth_digest.on'      : True,
    'tools.auth_digest.realm'   : db.auth_realm,
    'tools.auth_digest.get_ha1' : get_ha1,
    'tools.auth_digest.key'     : ''.join([random.choice(string.ascii_letters) for _ in range(64)]),
  }
}

class AIM_API(object):
    def __init__(self):
        self.face_core = core_api.EmotionRecognizer()

    def _cp_dispatch(self, vpath):
        if len(vpath) > 0 and vpath.pop(0) == 'api':
          if len(vpath) > 0 and vpath.pop(0) == 'face':
            return self.face_core

        return vpath


if __name__ == '__main__':
    #cherrypy.quickstart(Document(), '/api/document', config)
    cherrypy.quickstart(AIM_API(), '/', config)