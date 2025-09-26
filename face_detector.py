import cv2
import os
import glob
dataset_path = "FacialRecognitionProject/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cam = cv2.VideoCapture(0)
cam.set(3,640) #set video width
cam.set(4,480) #set video height
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
print("\n[INFO] Showing detected faces. pres 's' to start training...")
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xff
    if key == ord('s'):
        break
    elif key == 27: #press 'ESC' to stop
        cam.release()
        cv2.destroyAllWindows()
        exit()
        
face_id = input("\n enter user id end pres <return> ==>  ")

#Check how many images already exist for given user
existing_images = glob.glob(f"{dataset_path}/User.{face_id}.*.jpg")
existing_count = len(existing_images)
start_count = existing_count + 1
count = 0 #number of images in this session

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
#Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count +=1
        image_number = start_count + count - 1
        # save captured image into datasets folder
        filename = f"{dataset_path}/User.{face_id}.{image_number}.jpg"

        cv2.imwrite(filename, gray[y:y+h, x:x+w])
        cv2.imshow("image", img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 10: #take 10 face sample and stop video:
        break
#do cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
