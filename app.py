import cv2
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
import base64
import re
import cv2.face

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
    #DNN detection + LBPH recog
    if img is None or img.size == 0: 
        return jsonify({"processed_image": "","error":"Failed to decode image"}), 400
    if len(img.shape) != 3 or img.shape[2] != 3:
        return jsonify({"processed_image": "", "error":f"Image has incorrect channels ({img.shape}). Expected 3."}), 400
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h,w) = img.shape[:2]
    try:
        blob = cv2.dnn.blobFromImage(img,
                                     1.0,
                                     (300,300),
                                     (104.0,177.0,123.0),
                                     swapRB = False,
                                     crop=False
        )
        if blob is None or blob.size == 0:
            return jsonify({"processed_image": "", "error": "Blob creation resulted in empty blob"}), 500
    except Exception as e:
        return jsonify({"processed_image": "", "error": f"Blob creation failed: {e}"}), 500
    if face_detector:
        try:
            face_detector.setInput(blob)
            detections = face_detector.forward()
        except cv2.error as e:
            print(f"DNN forward pass error: {e}")
            return jsonify({"processed_image": "", "error": f"Processing error on forward pass. Input blob issue?"}), 500
        face_detector.setInput(blob)
        detections = face_detector.forward()
    else:
        detections = np.zeros((1,1,0,7))
    #Bounding Box logic
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY,endX,endY) = box.astype("int")
            startX = max(0,startX)
            startY = max(0,startY)
            endX = min(w, endX)
            endY = min(h, endY)

            w_face = endX-startX
            h_face = endY-startY

            x = startX
            y=startY

            cv2.rectangle(img,(int(x),int(y)),(int(endX),int(endY)),(0,255,0),2)
            face_roi = gray[y:y + h_face, x:x+w_face]
            if face_roi is None or face_roi.size == 0:
                continue
            try:
                face_roi_resized = cv2.resize(face_roi,(60,60))
            except cv2.error:
                continue
            id_num, confidence_lbph = recognizer.predict(face_roi_resized)
            threshold = 110
            name = names[id_num] if confidence_lbph < threshold else names[0]
            label = f"{name}: {round(confidence_lbph, 2)}"
            cv2.putText(img, label, (x+5,y-5),font,0.7,(0,255,0),2)

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

