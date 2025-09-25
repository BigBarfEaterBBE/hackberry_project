import cv2
bright = cv2.imread("bright_cube.jpg")
dark = cv2.imread("dark_cube.jpg")
#problems with RGB Color space:
# significant perpeutal non-uniformity
# mixing of color related information and intensity relation information data
#LAB Colorspace:
brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
cv2.imshow("brightLAB", brightLAB)
cv2.imshow("darkLAB", darkLAB)
cv2.waitKey(0)
cv2.destroyAllWindows()
#YCrCb Colorspace:
brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)
cv2.imshow("brightYCB", brightYCB)
cv2.imshow("darkYCB", darkYCB)
cv2.waitKey(0)
cv2.destroyAllWindows()
#HSV Colorspace:
brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
cv2.imshow("brightHSV", brightHSV)
cv2.imshow("darkHSV", darkHSV)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Applying threshold for segmentation
import numpy as np
bgr = [40, 158, 16]
thresh = 40
minBGR = np.array([bgr[0]-thresh,bgr[1]-thresh,bgr[2]-thresh])
maxBGR = np.array([bgr[0]+thresh,bgr[1]+thresh,bgr[2]+thresh])

maskBGR = cv2.inRange(bright,minBGR,maxBGR)
resultBGR = cv2.bitwise_and(bright,bright,mask = maskBGR)
#convert 1D array to 3D then convert it to HSV and take the first element
hsv = cv2.cvtColor(np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

minHSV = np.array([hsv[0]-thresh,hsv[1]-thresh,hsv[2]-thresh])
maxHSV = np.array([hsv[0]+thresh,hsv[1]+thresh,hsv[2]+thresh])
maskHSV = cv2.inRange(brightHSV, minHSV, maxHSV)
resultHSV = cv2.bitwise_and(brightHSV, brightHSV, mask = maskHSV)
#convert 1D array to 3D then convert to YCrCb and take the first elememt
ycb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]
minYCB = np.array([ycb[0]-thresh,ycb[1]-thresh,ycb[2]-thresh])
maxYCB = np.array([ycb[0]+thresh,ycb[1]+thresh,ycb[2]+thresh])
maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
resultYCB = cv2.bitwise_and(brightYCB, brightYCB, mask = maskYCB)
#convert 1D array to 3D then convert to LAB and take the first element
lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
minLAB = np.array([lab[0]-thresh,lab[1]-thresh,lab[2]-thresh])
maxLAB = np.array([lab[0]+thresh,lab[1]+thresh,lab[2]+thresh])
maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
resultLAB = cv2.bitwise_and(brightLAB, brightLAB, mask = maskLAB)
cv2.imshow("Result BGR", resultBGR)
cv2.imshow("Result HSV", resultHSV)
cv2.imshow("Result YCB", resultYCB)
cv2.imshow("Result LAB", resultLAB)
cv2.waitKey(0)
cv2.destroyAllWindows()