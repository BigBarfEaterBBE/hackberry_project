# Import dependencies
import cv2
# Read Images
img = cv2.imread('annotating_image.jpg')
# Display Image
cv2.imshow('Original Image',img)
cv2.waitKey(0)
# Print error message if image is null
if img is None:
    print('Could not read image')
# Draw line on image
imageLine = img.copy()
#line(image, start_point, end_point, color, thickness)
pointA = (200,80)
pointB = (450,80)
cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.imshow('Image Line', imageLine)
cv2.waitKey(0)
#draw circle
#circle(image, center_coordinates, radius, color, thickness)
imageCircle = img.copy()
#define circle center
circle_center = (415, 190)
#define radius
radius = 100
#draw a circle
cv2.circle(imageCircle, circle_center, radius, (0,0,255), thickness = 3, lineType = cv2.LINE_AA)
cv2.imshow("Image Circle", imageCircle)
cv2.waitKey(0)
#draw a filled circle
imageFilledCircle = img.copy()
circle_center = (415,190)
radius = 100
cv2.circle(imageFilledCircle, circle_center, radius, (255, 0,0), thickness = -1, lineType = cv2.LINE_AA)
cv2.imshow("Image with Filled Circle", imageFilledCircle)
cv2.waitKey(0)