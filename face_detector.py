#1 = ricky 2 = hanbin 3 = zhang hao 4 = yujin 5 = matt
#6 = jiwoong 7 = gunwook 8 = gyuvin 9 = taerae
import cv2
import os
import glob
import numpy as np
#Define paths for DL model
PROTO_PATH = "deploy.prototxt.txt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

#Init and setup
dataset_path = "FacialRecognitionProject/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

#Load DL face detector model
face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.5

cam = cv2.VideoCapture(0)
cam.set(3,640) #set video width
cam.set(4,480) #set video height

#face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

print("\n[INFO] Showing detected faces. pres 's' to start training...")

#Face detection loop(pre-training)

while True:
    ret, img = cam.read()
    (h,w) = img.shape[:2] #get image dimensions
    #convert image to blob for dl model
    #ssd expects 300x300 normalized input
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

    #perform face detection
    face_detector.setInput(blob)
    detections = face_detector.forward()

    #faces list store bound
    faces = []

    #loop over detections
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
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xff
    if key == ord('s'):
        break
    elif key == 27: #press 'ESC' to stop
        cam.release()
        cv2.destroyAllWindows()
        exit()
#data collection setup
face_id = input("\n enter user id end pres <return> ==>  ")

#Check how many images already exist for given user
existing_images = glob.glob(f"{dataset_path}/User.{face_id}.*.jpg")
existing_count = len(existing_images)
start_count = existing_count + 1
count = 0 #number of images in this session

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
#Initialize individual sampling face count
count = 0

#sampling loop
while(True):
    ret, img = cam.read()
    (h,w) = img.shape[:2]
    #convert image to blob
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0,177.0, 123.0))

    #perform face detection
    face_detector.setInput(blob)
    detections = face_detector.forward()
    faces = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            x = max(0,startX)
            y= max(0,startY)
            w_box = min(w-x, endX-startX)
            h_box = min(h-y,endY-startY)
            if w_box > 0 and h_box > 0:
                faces.append((x,y,w_box,h_box))
    #process and save
    for (x,y,w_box,h_box) in faces:
        cv2.rectangle(img,(x,y),(x+w_box,y+h_box),(0,255,0),2)

        #only save image if face detected
        count +=1
        image_number = start_count + count-1
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filename = f"{dataset_path}/User.{face_id}.{image_number}.jpg"
        cv2.imwrite(filename, gray_img[y:y+h_box, x:x+w_box])
        cv2.imshow("image",img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 20: #take 20 face sample and stop video:
        break
#do cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
