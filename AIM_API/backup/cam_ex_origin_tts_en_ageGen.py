#-*- coding: utf-8 -*-

import cv2
import numpy as np
import recognize_age
import recognize_gender
import tts_module

faceCascade = cv2.CascadeClassifier("face_profile/haarcascade_frontalface_alt_tree.xml")

video_capture = cv2.VideoCapture(0)
ttsModule = tts_module.tts_module()

Threadshold = 45
tts_AreaThreadhold=10000
tts_AreaThreadhold_NeedCal = 7000


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

ttsModule.start()

cnt = Threadshold - 60

cv2.namedWindow('ORIGINAL',cv2.WINDOW_NORMAL)
cv2.namedWindow('VIDEO',cv2.WINDOW_NORMAL)

cv2.resizeWindow('ORIGINAL',900,675)
cv2.resizeWindow('VIDEO',900,675)

cv2.moveWindow('ORIGINAL',100,100)
cv2.moveWindow('VIDEO',1000,100)

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
    maxAge=0
    maxGender=''
    maxGender_en=''
    maxGender_en_only=''
    (lx,ly,lw,lh) = (0,0,0,0)


    for (x, y, w, h) in faces:

        if(w*h > tts_AreaThreadhold_NeedCal):
            h_margin, w_margin = get_margin(x, y, w, h)
            crop = original_image[y-h_margin:y + h+h_margin,x-w_margin:x + w+w_margin]
            p = age_rec.recognize_age(crop)
            pg,pp = ged_rec.recognize_age(crop)

            if(w*h > tts_AreaThreadhold):
                cv2.rectangle(frame, (x-w_margin, y-h_margin), (x+w+w_margin, y+h+h_margin), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x - w_margin, y - h_margin), (x + w + w_margin, y + h + h_margin), (0, 0, 255), 2)



            cv2.putText(frame, "%.2f" % p, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            cv2.putText(frame, "%.2f" % p, (x - 1, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            age_integer = int(p)

            if pg == 1:
                ged = '남성'
                ged_en = 'male'
		ged_en_only = 'male'
            else:
                ged = '여성'
                ged_en = 'female'
		ged_en_only = 'female'

            ged_en += " (%.2f %%)"%(pp*100.)

            cv2.putText(frame, "%s" % ged_en, (x, y - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(frame, "%s" % ged_en, (x - 1, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            area = w * h
            if (maxRegion < area):
                maxRegion = area
                maxAge   = age_integer
                maxGender = ged
                maxGender_en=ged_en
		maxGender_en_only=ged_en_only
                (lx,ly,lw,lh)=(x,y,w,h)
        else:
            h_margin, w_margin = get_margin(x, y, w, h)
            cv2.rectangle(frame, (x - w_margin, y - h_margin), (x + w + w_margin, y + h + h_margin), (0, 0, 255), 2)


    if(maxRegion > tts_AreaThreadhold):
        if(cnt > Threadshold):
            ttsTxt = 'You look like ' +     maxGender_en_only + ', and ' + str(maxAge) + 'years old'
        # ttsModule.inputTxt(ttsTxt, 'en')
            ttsModule.inputTxt(ttsTxt, 'en')
            cnt=0
            h_margin, w_margin = get_margin(lx, ly, lw, lh)

            cv2.rectangle(original_image, (lx - w_margin, ly - h_margin), (lx + lw + w_margin, ly + lh + h_margin), (255, 0, 0), 3)


            cv2.putText(original_image, "%.2f" % maxAge, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(original_image, "%.2f" % maxAge, (lx - 1, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            cv2.putText(original_image, "%s" % maxGender_en, (lx, ly - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(original_image, "%s" % maxGender_en, (lx - 1, ly - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            cv2.imshow('ORIGINAL',original_image)

    cv2.imshow('VIDEO', frame)
    #cv2.imwrite('output/' + str(cnt) + '.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ttsModule.finishModule()
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
