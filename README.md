# hackberry_project
My project for hackberry!!
To familiarize myself with OpenCV, I used: https://learnopencv.com/getting-started-with-opencv/
For the face recognition I followed this tutorial for the framework: https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826#toc-step-5--data-gathering-7
But instead of using Haar cascades for recognition I used OpenCV deep learning facial recognition and added glob to count how many current files I had for a given userID so it would add more samples instead of replacing old ones
This is the repo I used for OpenCV deep learning facial recognition resources: https://github.com/alvareson/caffe-model-for-face-detection?utm_source=chatgpt.com

SETUP:
1) You need to download index.html, deploy.prototxt.txt, res10_300x300_ssd_iter_140000.caffemodel, trainer.py, and the dataset folder within the FacialRecognitionProject folder
2) If you want to host it on the browser download app.py and if you're just using it through your code editor, download recognizer.py
3) To get the model itself, run trainer.py, it will take a few seconds to download the model file
FOR WEB HOSTING VER. ONLY:
4) When you run app.py it will print: * Server running on: http://10.0.0.103:5000/ * Use this URL on your browser.
5) If you are having trouble with the link it provides, use http://localhost:5000/ *ONLY WORKS IF YOU ARE OPENING IT ON THE SAME DEVICE AS YOU RAN THE PROGRAM ON* 

NOTE:
I spent like 2 extra hours trying to implement SFace or Facenet models instead of LBPH and it lowk would not work so i made it into seperate branch but for some reason its like not there so idk what the jelly happened to it but thats why it looks like theres a big gap in commits