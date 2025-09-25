'''
you can rotate an image by a certain angle theta by defining a transformation matrix
M = [cos theta  -sin theta]
    [sin theta  cos theta]
OpenCV also provide ability for center of ratation + scaling
[a  B   (1-a)*cx-B*cy]
[-B a   B*cx+(1-a)*cy]
a = scale * cos theta
b = scale * sin theta
cx & cy = center of rotation
OpenCV gives getRotationMatrix2D() to create the transformation matrix
getRotationMatrix2D(center, angle, scale)
TO ROTATE:
1. get the center of rotation
2. create the 2D-rotation matrix
3. apply affine transformation to the image, using rotation matrix use warpAffine()
warpAffine(src, M, dszize[, dst[, flags[, borderMode[, borderValue]]]])
src = source image
M = transformation matrix
dsize = size of output image
dst = output image
flags = interpolation method such as INTER_LINEAR or INTER_NEAREST
borderMode = pixel extrapolation method
borderValue = value used in case of a constant border
'''
import cv2

#reading image
image = cv2.imread("translation+rotation_image.png")
#Dividing height + width by 2 to get center of image
height, width = image.shape[:2]
center = (width/2, height/2)
#use cv2.getRotationMatrix2D() to get rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle = 45, scale = 1)
#rotate image using cv2.warpAffine()
rotated_image = cv2.warpAffine(src = image, M=rotate_matrix, dsize = (width,height))
#visualize original + rotated image
cv2.imshow("Original Image", image)
cv2.imshow("Rotated iamge", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Image Translation
pixels by which the image needs to be shifted = tx + ty
M = [1 0 tx]
    [0 1 ty]
- Providing positive values of tx will shift to right and vice versa
- Providing positive values of ty will shift down and vice versa
TO TRANSLATE:
1. Read image + obtain width & height
2. create a transformation matrix
3. use warpAffine()
'''
import numpy as np
#read image
image = cv2.imread("translation+rotation_image.png")
#get width + height
height, width = image.shape[:2]
#get tx + ty
#can be any value of your choice
tx, ty = width/4, height/4
#create translation matrix using tx and ty, it is a NumPy array
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)
#apply translation to image
translated_image = cv2.warpAffine(src = image, M=translation_matrix, dsize=(width,height))
#display original + translated image
cv2.imshow("translated iamge", translated_image)
cv2.imshow("original image", image)
cv2.waitKey(0)
