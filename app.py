import cv2
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
import base64
import re

app = Flask(__name__)
#--Global Model Setup
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
PROTO_PATH = "deploy.prototxt.txt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
names = ["Human", "Ricky", "Hanbin", "Zhang Hao", "Yujin", "Matthew", "Jiwoong", "Gunwook", "Gyuvin", "Taerae"]

#FONTEND
@app.route("/")
def index():
    #renders HTML page
    return render_template("index.html")
@app.route("/process_frame",methods=["POST"])
def process_frame():
    #1 recieve Base64 image data from browser
    data = request.json["image"]
    #remove data URI header
    img_data = re.sub("^data:image/.+;base64,","",data)
    #convert Base65 string to numpy array
    try:
        nparr = np.frombuffer(base64.b64decode(img_data),np.uint8)
        img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    except:
        return jsonify({"processed_image":"","error":"Invalid image data"}),400
    #2 run recognition logic
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h,w) = img.shape[:2]
    #DNN detection + LBPH recog
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)),1.0,(300,300),(104.0,177.0,123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    faces = []
    #Bounding Box logic
    for (x,y,w_face,h_face) in faces:
        cv2.rectangle(img,(x,y),(x+w_face,y+h_face),(0,255,0),2)
        face_roi = gray[y:y + h_face, x:x + w_face]
        try:
            face_roi_resized = cv2.resize(face_roi,(60,60)) #use trained size
        except cv2.error:
            continue
        id_num, confidence = recognizer.predict(face_roi_resized)
        threshold = 70
        name = names[id_num] if confidence < threshold else names[0]

        cv2.putText(img, str(name), (x+5, y-5), font, 1, (255,255,255),2)
    #3 encode processed image back to Base64
    _, buffer = cv2.imencode(".jpeg", img)
    processed_base64 = base64.b64encode(buffer).decode("utf-8")
    #4 return processed image base64 string
    return jsonify({"processed_image":processed_base64})

if __name__ == "__main__":
    #(ip adresses retrieval)
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8",80))
    local_ip = s.getsockname()[0]
    s.close()

    print(f"\n* Server running on: http://{local_ip}:5000/")
    print(f"* Use this URL on iPhone browser.\n")
    app.run(host="0.0.0.0", port = 5000, debug = False, threaded = True)

