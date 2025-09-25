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

#draw a rectangle
#rectangle(image, start_point, end_point, color, thickness)
#starting_point = top left, ending_point = bottom right
imageRectangle = img.copy()
start_point = (300,115)
end_point=(475,225)
cv2.rectangle(imageRectangle, start_point, end_point, (0,0,255), thickness = 3, lineType = cv2.LINE_AA)
cv2.imshow("Image Rectangle", imageRectangle)
cv2.waitKey(0)
#draw an ellipse
#ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color, thickness)
imageEllipse = img.copy()
ellipse_center = (415,190)
axis1 = (100,50)
axis2 = (125, 50)
cv2.ellipse(imageEllipse, ellipse_center, axis1, 0, 0, 360, (255,0,0), thickness = 3)
cv2.ellipse(imageEllipse, ellipse_center, axis2, 90, 0, 360, (0,0,255), thickness = 3)
cv2.imshow("ellipse image", imageEllipse)
cv2.waitKey(0)
#draw a half-ellipse
#to do this: set endAngle = 180
halfEllipse = img.copy()
#use same ellipse center as above
cv2.ellipse(halfEllipse, ellipse_center, axis1, 0, 180, 360, (255,0,0), thickness = 3)
#to draw a filled ellipse:
cv2.ellipse(halfEllipse, ellipse_center, axis1, 0,0,180, (0,0,255), thickness = -2)
cv2.imshow("halfEllipse",halfEllipse)
cv2.waitKey(0)
#Adding text
#putText(image, text, org, font, fontScale, color)
#org = starting location for top left of text string
#OpenCV supports several font-face styles from Hershey font collection
'''FONT_HERSHEY_SIMPLEX        = 0,
   FONT_HERSHEY_PLAIN          = 1,
   FONT_HERSHEY_DUPLEX         = 2,
   FONT_HERSHEY_COMPLEX        = 3,
   FONT_HERSHEY_TRIPLEX        = 4,
   FONT_HERSHEY_COMPLEX_SMALL  = 5,
   FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
   FONT_HERSHEY_SCRIPT_COMPLEX = 7,
   FONT_ITALIC                 = 16'''
imageText = img.copy()
text = "I am a Happy dog!"
org = (50,350)
cv2.putText(imageText, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (250,225,100))
cv2.imshow("Image Text", imageText)
cv2.waitKey(0)

cv2.destroyAllWindows()