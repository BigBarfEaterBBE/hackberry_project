import cv2
import numpy as np
from PIL import Image
import os
# DELETE/COMMENT OUT: PROTO_PATH, MODEL_PATH, detector, CONFIDENCE_THRESHOLD

path = "FacialRecognitionProject/dataset" # Ensure this path is correct

recognizer = cv2.face.LBPHFaceRecognizer.create() # Keep this line

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    # 💡 Simplified Logic: Loop through pre-cropped images
    for imagePath in imagePaths:
        # 1. Load the image directly as grayscale
        # The images you saved are already cropped, just load them.
        PIL_img = Image.open(imagePath).convert("L") 
        img_numpy = np.array(PIL_img, "uint8")
        
        # Check if the image array is empty (which can still happen if a file is corrupted)
        if img_numpy.size == 0 or img_numpy.shape[0] < 1 or img_numpy.shape[1] < 1:
            print(f"Skipping empty or invalid image file: {imagePath}")
            continue

        # 2. Extract the ID from the filename
        # This assumes filenames are correct: User.ID.ImageNumber.jpg
        try:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        except (IndexError, ValueError):
            print(f"Skipping file with invalid name format: {imagePath}")
            continue

        # 3. Add the loaded image directly to the samples
        faceSamples.append(img_numpy)
        ids.append(id)
        
    return faceSamples, ids

# Keep the rest of your training script:
print("\n [INFO] Training faces. It will take a few seconds. Wait...")
faces,ids = getImagesAndLabels(path)

# Add a final check to ensure lists are not empty, in case all files were invalid
if len(faces) == 0:
    print("\n [FATAL ERROR] No valid face samples found in the dataset folder. Check path and file contents.")
else:
    # This line should now work without the cv2.error
    recognizer.train(faces, np.array(ids)) 
    recognizer.write("trainer.yml")
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))