import requests
from requests.auth import HTTPDigestAuth
from requests.packages.urllib3.exceptions import InsecureRequestWarning

"""
Python 2.7 client code example for gender/age/emotion recognition api
"""

# to disable SSL warning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

with open('./test_vid/angry.mp4', 'rb') as f:
    vid_bytes = f.read()  # read file as byte-array format

# authentication protocol: HTTP Digest. ID and Password will be needed
auth = HTTPDigestAuth('flagship', 'flagship')

# open request session
s = requests.Session()

# fetch a result (note that the request is in octet-stream (byte array))
r = s.post('https://143.248.136.158:8080/api/face', auth=auth,
       headers={'Content-Type': 'application/octet-stream'}, data=vid_bytes, verify=False)

# check response status and headers
print r.status_code, r.headers

# if the status OK,
if r.status_code == 200 and r.headers['Content-Type'] == 'application/json':
    print r.content  # we can check the response by printing it out

