import cv2
#cv2.VideoCapture(path, apiPreference) path = filename
vid_capture = cv2.VideoCapture('image.gif')
if(vid_capture.isOpened()==False):
    print("Error opening the video file")
else:
    #get frame rate information
    fps = int(vid_capture.get(5)) #5 = CAP_PROP_FPS = frame rate
    print("Frame Rate : ", fps)
    #get frame count
    frame_count = vid_capture.get(7) #7 = CAP_PROP_FRAME_COUNT = frame count
    print("Frame count : ", frame_count)
#vid_capture.read() returns a tuple (bool if there is a frame to read, video frame)
while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(20) #113 is ASCII code for q key
        if k == 113:
            break
    else:
        break
#release objects
vid_capture.release()
cv2.destroyAllWindows()
# to write a video file:
# retrieve frame height + width using get()
# init video capture object to read vid stream into memory
# create a video writer object
# use video writer object to save video stream to disk
vid_capture = cv2.VideoCapture('image.gif')
#Obtain frame size info using get()
frame_width = int(vid_capture.get(3)) #3 = CAP_PROP_FRAME_WIDTH
frame_height = int(vid_capture.get(4)) #4 = CAP_PROP_FRAME_HEIGHT
frame_size = (frame_width, frame_height)
fps = 20
#syntax for VideoWriter(): VideoWriter(filename, apiPreference, fourcc, fps, framSize[,isColor])
#file name: pathname for output file, apiPreference: API backends identifier, fourcc: 4-character code used to compress frames, fps: frame rate, frame_size: size of the video frames, isColor: if not zero, encoder will expect and encode color frames
output = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
        #wrte the frame to the output file
        output.write(frame)
    else:
        print('Stream disconnected')
        break
#release objects
vid_capture.release()   
output.release()
