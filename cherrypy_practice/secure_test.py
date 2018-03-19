#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cherrypy


userpassdict  = {'user1': 'passwd'}
checkpassword = cherrypy.lib.auth_basic.checkpassword_dict(userpassdict)
get_ha1       = cherrypy.lib.auth_digest.get_ha1_dict_plain(userpassdict)

config = {
  'global' : {
    'server.socket_host' : '127.0.0.1',
    'server.socket_port' : 8080,
    'server.thread_pool' : 8
  },
  '/' : {
    # HTTP verb dispatcher
    'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
    # JSON response
    'tools.json_out.on' : True,
    # Basic Auth
    #'tools.auth_basic.on'            : True,
    #'tools.auth_basic.realm'         : 'Walled garden',
    #'tools.auth_basic.checkpassword' : checkpassword,
    # Digest Auth
    'tools.auth_digest.on'      : True,
    'tools.auth_digest.realm'   : 'Walled garden',
    'tools.auth_digest.get_ha1' : get_ha1,
    'tools.auth_digest.key'     : 'put random secret here',
  }
}


class Document:
  '''Test like:
  curl --user user1:passwd --request GET http://localhost:8080/api/document
  curl --user user1:passwd --request GET http://localhost:8080/api/document/2
  curl --user user1:passwd --request POST --data name="new entry" http://localhost:8080/api/document
  curl --user user1:passwd --request PUT --data name="new entry2" http://localhost:8080/api/document/4
  curl --user user1:passwd --request DELETE http://localhost:8080/api/document/4
  '''

  _store  = None
  exposed = True


  def __init__(self):
    self._store = {
      1 : {'id': 1, 'name': 'foo'},
      2 : {'id': 2, 'name': 'bar'},
      3 : {'id': 3, 'name': 'baz'},
      4 : {'id': 4, 'name': 'qux'},
    }

  def GET(self, id = None):
    if id:
      return self._store[int(id)]
    else:
      return self._store.values()

  def POST(self, **kwargs):
    id = max(self._store.keys()) + 1
    self._store[id] = {'id': id, 'name': kwargs['name']}
    return id

  def PUT(self, id, **kwargs):
    self._store[int(id)].update(kwargs)

  def DELETE(self, id):
    self._store.pop(int(id))


if __name__ == '__main__':
  cherrypy.quickstart(Document(), '/api/document', config)