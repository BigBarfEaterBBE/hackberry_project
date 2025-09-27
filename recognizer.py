import cv2
import numpy as np
import os
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
PROTO_PATH = "deploy.prototxt.txt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ["Human", "Ricky", "Hanbin", "Zhang Hao", "Yujin", "Matthew", "Jiwoong", "Gunwook", "Gyuvin", "Taerae"]
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h,w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        #extract confidence
        confidence = detections[0,0,i,2]

        #filter out weak detections by ensuring cofindence is greater than min thresh
        if confidence> CONFIDENCE_THRESHOLD:
            #compute (x,y)-coordinates of bounding box for face
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")

            #store bounding box as (x,y,w,h) for consistency with the original code
            #ensure coords are within frame bound
            x = max(0, startX)
            y = max(0, startY)
            w_box = min(w-x, endX- startX)
            h_box = min(h-y, endY-startY)

            #only append if box has positive dimensions
            if w_box > 0 and h_box > 0:
                faces.append((x,y,w_box, h_box))
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        face_roi = gray[y:y+h,x:x+w]
        id, confidence = recognizer.predict(face_roi)
        if (confidence < 70):
            id = names[id]
        else:
            id = names[0]
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255),2)
    cv2.imshow("camera", img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()