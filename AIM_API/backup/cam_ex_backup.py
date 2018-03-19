#-*- coding: utf-8 -*-

import cv2
import numpy as np
import recognize_age
import recognize_gender
import tts_module

faceCascade = cv2.CascadeClassifier("face_profile/haarcascade_frontalface_alt_tree.xml")

video_capture = cv2.VideoCapture(0)
ttsModule = tts_module.tts_module()

def get_margin(x,y,w,h):
    h_margin = h // 8
    w_margin = w // 8
    if y + h >= height:
        h_margin = height - y - 1
    if x + w >= width:
        w_margin = width - x - 1
    if y - h_margin < 0:
        h_margin = y - 1
    if x - w_margin < 0:
        w_margin = x - 1
    return h_margin, w_margin

age_rec = recognize_age.runner()
ged_rec = recognize_gender.runner()

cnt = 0

ttsModule.start()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    cnt += 1
    original_image = np.array(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE

    )


    # Draw a rectangle around the faces
    crop_image = []
    height , width = original_image.shape[:2]

    maxRegion = -9999
    maxx = -1
    maxy = -1
    maxw = -1
    maxh = -1

    for(x,y,w,h) in faces:
        area = w * h
        if(area > maxRegion):
            (maxx,maxy,maxw,maxh) = (x,y,w,h)
            maxRegion = area


    if(maxx<0):
        continue

    (x,y,w,h) = (maxx, maxy, maxw, maxh)

    # for (x,y,w,h) in faces:
    h_margin, w_margin = get_margin(x, y, w, h)
    crop = original_image[y-h_margin:y + h+h_margin,x-w_margin:x + w+w_margin]
    p = age_rec.recognize_age(crop)
    pg, pp = ged_rec.recognize_age(crop)
    cv2.rectangle(frame, (x-w_margin, y-h_margin), (x+w+w_margin, y+h+h_margin), (0, 255, 0), 2)

    cv2.putText(frame, "%.2f" % p, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    cv2.putText(frame, "%.2f" % p, (x - 1, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    age_integer = int(p)

    if pg == 1:
        ged = '남성'
        ged_en = 'male'
    else:
        ged = '여성'
        ged_en = 'female'



    cv2.putText(frame, "%s" % ged_en, (x, y - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(frame, "%s (%s)" % (ged_en, pp), (x - 1, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    ttsTxt = '성별은 ' + ged + '나이는 ' + str(age_integer) + '세'
    #ttsModule.inputTxt(ttsTxt, 'en')
    ttsModule.inputTxt(ttsTxt, 'ko')



    # Display the resulting frame
    cv2.imshow('ORIGINAL',original_image)
    cv2.imshow('Video', frame)
    #cv2.imwrite('output/' + str(cnt) + '.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ttsModule.finishModule()
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
