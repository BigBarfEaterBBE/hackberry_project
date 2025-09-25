import cv2
cap = cv2.VideoCapture(0)
#USES HAAR CASCADE CLASSIFIER TO DETECT FACES
faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
cap.set(3,640)#width
cap.set(4,480)#height
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5, minSize = (20,20))
    for (x,y,w,h,) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow("video",img)
    k = cv2.waitKey(3) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()