import cv2
import numpy as np
from PIL import Image
import os
# DELETE/COMMENT OUT: PROTO_PATH, MODEL_PATH, detector, CONFIDENCE_THRESHOLD

path = "FacialRecognitionProject/dataset"
BATCH_SIZE = 75
recognizer = cv2.face.LBPHFaceRecognizer_create() 

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    #Simplified Logic: Loop through pre-cropped images
    for imagePath in imagePaths:
        # Load the image directly as grayscale
        PIL_img = Image.open(imagePath).convert("L")
        PIL_img = PIL_img.resize((60, 60), Image.Resampling.LANCZOS)
        img_numpy = np.array(PIL_img, "uint8")

        
        # Check if the image array is empty
        if img_numpy.size == 0 or img_numpy.shape[0] < 1 or img_numpy.shape[1] < 1:
            print(f"Skipping empty or invalid image file: {imagePath}")
            continue

        # 2. Extract the ID from the filename
        try:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        except (IndexError, ValueError):
            print(f"Skipping file with invalid name format: {imagePath}")
            continue

        # 3. Add the loaded image directly to the samples
        faceSamples.append(img_numpy)
        ids.append(id)
        
    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait...")
faces,ids = getImagesAndLabels(path)

# Add a final check to ensure lists are not empty, in case all files were invalid
if len(faces) == 0:
    print("\n [FATAL ERROR] No valid face samples found in the dataset folder. Check path and file contents.")
else:
    total_samples = len(faces)
    print(f"\n [INFO] Total samples loaded: {total_samples}")
    print(f" [INFO] Training using batches of size: {BATCH_SIZE}")
    is_first_batch = True
    for i in range(0,total_samples, BATCH_SIZE):
        start_index = i
        end_index = min(i+ BATCH_SIZE, total_samples)
        #extract current batch data
        batch_faces = faces[start_index:end_index]
        batch_ids = np.array(ids[start_index:end_index])
        if len(batch_faces) == 0:
            continue
        print(f" [INFO] Processing batch from index {start_index} to {end_index-1}...")
        if is_first_batch:
            #1 use .train() for first batch to init model
            recognizer.train(batch_faces,batch_ids)
            is_first_batch = False
            print(" [INFO] Initial training completed.")
        else:
            #2 use .update() for subsequent to incrementally refine model
            recognizer.update(batch_faces,batch_ids)
            print(" [INFO] Model updated successfully.")

    recognizer.write("trainer.yml")
    print("\n [INFO] Training complete. Model saved to trainer.yml")
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))